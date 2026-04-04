#!/usr/bin/env python3
"""
Per-rank Python worker for distributed (multi-card) kernel execution.

Replaces the monolithic C++ distributed_worker binary.  Each rank runs
as a separate process, using the comm_* C API (via ctypes bindings) for
HCCL / sim communication and the existing PTO runtime C API for kernel
execution.

Spawned by DistributedCodeRunner — not intended for direct invocation.
"""

import argparse
import struct
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "python"))
sys.path.insert(0, str(script_dir))


DTYPE_FORMAT = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint32": ("I", 4),
    "uint64": ("Q", 8),
    "float16": ("e", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "bfloat16": ("H", 2),
}


def parse_buffer_spec(spec):
    parts = spec.split(":")
    result = {"name": parts[0], "dtype": parts[1], "count": int(parts[2])}
    if len(parts) >= 4:
        result["data_prefix_elems"] = int(parts[3])
    else:
        result["data_prefix_elems"] = 0
    if len(parts) >= 5 and parts[4]:
        result["shape"] = [int(dim) for dim in parts[4].split(",") if dim]
    else:
        result["shape"] = [result["count"]]
    return result


def parse_kernel_spec(spec):
    p = spec.index(":")
    return {"func_id": int(spec[:p]), "filename": spec[p + 1:]}


def parse_phase_spec(spec):
    parts = spec.split(":")
    barrier_before = False
    if len(parts) >= 2:
        barrier_before = bool(int(parts[1]))
    args = None
    if len(parts) >= 3 and parts[2]:
        args = []
        for item in parts[2].split(","):
            if not item:
                continue
            token, kind = item.split("@", 1)
            args.append({"token": token, "kind": kind})
    return {"orch_func": parts[0], "barrier_before": barrier_before, "args": args}


def main():
    parser = argparse.ArgumentParser(description="Distributed per-rank worker")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nranks", type=int, required=True)
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--rootinfo-file", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--orch-file", required=True)
    parser.add_argument("--orch-func", default=None)
    parser.add_argument("--phase2-orch-func", default=None)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--win-sync-prefix", type=int, default=0)
    parser.add_argument("--aicpu-thread-num", type=int, default=1)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--orch-thread-num", type=int, default=0)
    parser.add_argument("--win-buffer", action="append", default=[])
    parser.add_argument("--dev-buffer", action="append", default=[])
    parser.add_argument("--load", action="append", default=[], dest="loads")
    parser.add_argument("--save", action="append", default=[], dest="saves")
    parser.add_argument("--arg", action="append", default=[], dest="args")
    parser.add_argument("--kernel-bin", action="append", default=[])
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    data_dir = Path(args.data_dir) if args.data_dir else artifact_dir / f"rank_{args.rank}"

    buffers = []
    for spec in args.win_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "window"
        buffers.append(b)
    for spec in args.dev_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "device"
        buffers.append(b)

    kernel_bins = [parse_kernel_spec(s) for s in args.kernel_bin]
    phases = [parse_phase_spec(s) for s in args.phase]
    if not phases:
        if args.orch_func:
            phases.append({"orch_func": args.orch_func, "barrier_before": False})
        if args.phase2_orch_func:
            phases.append({"orch_func": args.phase2_orch_func, "barrier_before": True})
    if not phases:
        sys.stderr.write(f"[rank {args.rank}] no orchestration phases configured\n")
        return 1

    buf_by_name = {b["name"]: b for b in buffers}

    def elem_size(dtype):
        return DTYPE_FORMAT.get(dtype, ("f", 4))[1]

    def buf_prefix_bytes(b):
        return b.get("data_prefix_elems", 0) * elem_size(b["dtype"])

    def buf_bytes(b):
        return b["count"] * elem_size(b["dtype"])

    # ----------------------------------------------------------------
    # 1. Load library
    # ----------------------------------------------------------------
    from bindings import (
        bind_host_binary, set_device,
        device_malloc, device_free, copy_to_device, copy_from_device,
        comm_init, comm_alloc_windows, comm_get_local_window_base,
        comm_barrier, comm_destroy,
    )
    from task_interface import (
        CallConfig,
        ChipCallable,
        ChipStorageTaskArgs,
        ChipWorker,
        ContinuousTensor,
        CoreCallable,
        DataType,
    )

    lib_path = artifact_dir / "libhost_runtime.so"
    bind_host_binary(str(lib_path))
    sys.stderr.write(f"[rank {args.rank}] Library loaded\n")

    # ----------------------------------------------------------------
    # 2. Comm init + alloc windows
    # ----------------------------------------------------------------
    comm = comm_init(args.rank, args.nranks, args.device_id, args.rootinfo_file)

    total_win = args.win_sync_prefix
    for b in buffers:
        if b["placement"] == "window":
            total_win += buf_prefix_bytes(b) + buf_bytes(b)

    device_ctx_ptr = comm_alloc_windows(comm, total_win)
    local_base = comm_get_local_window_base(comm)

    sys.stderr.write(f"[rank {args.rank}] Comm initialized, local_base=0x{local_base:x}\n")

    set_device(args.device_id)
    sys.stderr.write(f"[rank {args.rank}] Device {args.device_id} set for runtime\n")

    if args.win_sync_prefix > 0:
        import ctypes

        zero_bytes = bytes(args.win_sync_prefix)
        zero_buf = (ctypes.c_uint8 * len(zero_bytes)).from_buffer_copy(zero_bytes)
        copy_to_device(local_base, ctypes.addressof(zero_buf), len(zero_bytes))

    # ----------------------------------------------------------------
    # 3. Allocate buffers
    # ----------------------------------------------------------------
    win_offset = args.win_sync_prefix

    for b in buffers:
        nbytes = buf_bytes(b)
        prefix_bytes = buf_prefix_bytes(b)
        if b["placement"] == "window":
            win_offset += prefix_bytes
            b["dev_ptr"] = local_base + win_offset
            win_offset += nbytes
        else:
            ptr = device_malloc(prefix_bytes + nbytes)
            if not ptr:
                sys.stderr.write(f"[rank {args.rank}] device_malloc failed for '{b['name']}'\n")
                return 3
            b["alloc_ptr"] = ptr
            b["dev_ptr"] = ptr + prefix_bytes
        sys.stderr.write(
            f"[rank {args.rank}] Buffer '{b['name']}': {b['placement']} "
            f"{b['count']}x{b['dtype']}={nbytes}B"
            f" prefix={prefix_bytes}B @ 0x{b['dev_ptr']:x}\n"
        )

    # ----------------------------------------------------------------
    # 4. Load inputs
    # ----------------------------------------------------------------
    for name in args.loads:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --load: buffer '{name}' not found\n")
            return 1
        path = data_dir / f"{name}.bin"
        host_data = path.read_bytes()
        if len(host_data) != buf_bytes(b):
            sys.stderr.write(
                f"[rank {args.rank}] Size mismatch for '{name}': "
                f"file={len(host_data)}, expected={buf_bytes(b)}\n"
            )
            return 2
        import ctypes
        host_buf = (ctypes.c_uint8 * len(host_data)).from_buffer_copy(host_data)
        copy_to_device(b["dev_ptr"], ctypes.addressof(host_buf), len(host_data))

    # ----------------------------------------------------------------
    # 5. Barrier before kernel execution
    # ----------------------------------------------------------------
    comm_barrier(comm)

    # ----------------------------------------------------------------
    # 6. Run simpler runtime
    # ----------------------------------------------------------------
    orch_binary = (artifact_dir / args.orch_file).read_bytes()
    aicpu_binary = (artifact_dir / "libaicpu_kernel.so").read_bytes()
    aicore_binary = (artifact_dir / "aicore_kernel.o").read_bytes()

    kernel_binaries = []
    for k in kernel_bins:
        data = (artifact_dir / k["filename"]).read_bytes()
        kernel_binaries.append((k["func_id"], data))

    dtype_map = {
        "float32": DataType.FLOAT32,
        "float64": DataType.FLOAT32,
        "float16": DataType.FLOAT16,
        "bfloat16": DataType.BFLOAT16,
        "int8": DataType.INT8,
        "uint8": DataType.UINT8,
        "int16": DataType.INT16,
        "uint16": DataType.INT16,
        "int32": DataType.INT32,
        "uint32": DataType.INT32,
        "int64": DataType.INT64,
        "uint64": DataType.UINT64,
    }

    def resolve_scalar_token(tok):
        if tok == "nranks":
            return args.nranks
        if tok == "root":
            return args.root
        if tok == "deviceCtx":
            return device_ctx_ptr
        b = buf_by_name.get(tok)
        if b is None:
            raise KeyError(tok)
        return b["dev_ptr"]

    def build_phase_args(phase):
        phase_args = phase.get("args")
        if not phase_args:
            phase_args = [{"token": tok, "kind": "scalar"} for tok in args.args]

        orch_args = ChipStorageTaskArgs()
        seen_scalar = False
        tensor_count = 0
        scalar_count = 0
        for entry in phase_args:
            token = entry["token"]
            kind = entry.get("kind", "scalar")
            if kind == "tensor":
                if seen_scalar:
                    raise ValueError(f"Tensor arg '{token}' appears after scalar args")
                b = buf_by_name.get(token)
                if b is None:
                    raise KeyError(token)
                dtype = dtype_map.get(b["dtype"])
                if dtype is None:
                    raise ValueError(f"Unsupported tensor dtype '{b['dtype']}' for '{token}'")
                shape = tuple(int(dim) for dim in b.get("shape", [b["count"]]))
                orch_args.add_tensor(ContinuousTensor.make(b["dev_ptr"], shape, dtype))
                tensor_count += 1
            else:
                seen_scalar = True
                orch_args.add_scalar(resolve_scalar_token(token))
                scalar_count += 1
        return orch_args, tensor_count, scalar_count

    def run_phase(orch_func_name: str) -> None:
        phase = next((item for item in phases if item["orch_func"] == orch_func_name), None)
        if phase is None:
            raise ValueError(f"Unknown phase '{orch_func_name}'")
        orch_args, tensor_count, scalar_count = build_phase_args(phase)
        sys.stderr.write(
            f"[rank {args.rank}] Launching kernel phase '{orch_func_name}': "
            f"{tensor_count} tensors, {scalar_count} scalars, "
            f"{len(kernel_binaries)} kernels\n"
        )

        core_children = []
        for func_id, binary in kernel_binaries:
            core_children.append((func_id, CoreCallable.build(signature=[], binary=binary)))
        chip_callable = ChipCallable.build(
            signature=[],
            func_name=orch_func_name,
            binary=orch_binary,
            children=core_children,
        )

        worker = ChipWorker()
        worker.init(args.device_id, lib_path, aicpu_binary, aicore_binary)

        config = CallConfig()
        config.block_dim = args.block_dim
        config.aicpu_thread_num = args.aicpu_thread_num
        config.orch_thread_num = args.orch_thread_num
        worker.run(chip_callable, orch_args, config)
        worker.reset()

    for idx, phase in enumerate(phases):
        if idx > 0 and phase.get("barrier_before", False):
            comm_barrier(comm)
        run_phase(phase["orch_func"])
    sys.stderr.write(f"[rank {args.rank}] Kernel execution complete\n")

    # ----------------------------------------------------------------
    # 7. Barrier + save outputs
    # ----------------------------------------------------------------
    comm_barrier(comm)

    import ctypes
    for name in args.saves:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --save: buffer '{name}' not found\n")
            continue
        nbytes = buf_bytes(b)
        host_buf = (ctypes.c_uint8 * nbytes)()
        copy_from_device(ctypes.addressof(host_buf), b["dev_ptr"], nbytes)
        path = data_dir / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(host_buf))
        sys.stderr.write(f"[rank {args.rank}] Saved '{name}' to {path} ({nbytes}B)\n")

    # ----------------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------------
    for b in buffers:
        if b["placement"] == "device" and b.get("alloc_ptr"):
            device_free(b["alloc_ptr"])

    comm_destroy(comm)
    sys.stderr.write(f"[rank {args.rank}] Done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
