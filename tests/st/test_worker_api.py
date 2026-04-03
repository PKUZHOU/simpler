#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ST: Worker API end-to-end on sim platform.

Case 1: Worker(level=2) runs vector_example.
Case 2: Worker(level=3) runs ChipTask → SubWorkerTask (dependency via TensorMap).
"""

import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "scripts"))

import importlib.util  # noqa: E402
from multiprocessing.shared_memory import SharedMemory  # noqa: E402

import torch  # noqa: E402
from kernel_compiler import KernelCompiler  # noqa: E402
from task_interface import (  # noqa: E402
    ChipCallable,
    ChipStorageTaskArgs,
    CoreCallable,
    WorkerPayload,
    WorkerType,
    make_tensor_arg,
)
from worker import Task, Worker  # noqa: E402

# ---------------------------------------------------------------------------
# Compile kernels (common)
# ---------------------------------------------------------------------------

PLATFORM = "a2a3sim"
RUNTIME = "tensormap_and_ringbuffer"
KERNELS_DIR = ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
PTO_ISA = ROOT / "examples/scripts/_deps/pto-isa"

spec = importlib.util.spec_from_file_location("kconf", KERNELS_DIR / "kernel_config.py")
kconf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kconf)

print(f"[{time.time():.0f}] Compiling kernels...", flush=True)
kc = KernelCompiler(PLATFORM)
inc_dirs = kc.get_orchestration_include_dirs(RUNTIME)
orch_bin = kc.compile_orchestration(RUNTIME, str(kconf.ORCHESTRATION["source"]), extra_include_dirs=inc_dirs)
children = []
for k in kconf.KERNELS:
    bin_o = kc.compile_incore(
        str(k["source"]), core_type=k["core_type"], pto_isa_root=str(PTO_ISA), extra_include_dirs=inc_dirs
    )
    cc = CoreCallable.build(k.get("signature", []), bin_o)
    children.append((k["func_id"], cc))
CHIP_CALLABLE = ChipCallable.build(
    kconf.ORCHESTRATION.get("signature", []),
    kconf.ORCHESTRATION["function_name"],
    orch_bin,
    children,
)
CFG = kconf.RUNTIME_CONFIG
print(f"[{time.time():.0f}] Compiled OK", flush=True)


def make_tensors():
    SIZE = 128 * 128
    a = torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()
    b = torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()
    f = torch.zeros(SIZE, dtype=torch.float32).share_memory_()
    args = ChipStorageTaskArgs()
    for t in [a, b, f]:
        args.add_tensor(make_tensor_arg(t))
    return a, b, f, args


# ---------------------------------------------------------------------------
# Case 1: Worker(level=2)
# ---------------------------------------------------------------------------


def test_case1():
    print("\n" + "=" * 50, flush=True)
    print("Case 1: Worker(level=2, device_id=0) — vector_example", flush=True)
    print("=" * 50, flush=True)

    a, b, f, orch_args = make_tensors()

    w = Worker(level=2, device_id=0, platform=PLATFORM, runtime=RUNTIME)
    w.init()
    print(f"[{time.time():.0f}] Worker init OK", flush=True)

    w.run(CHIP_CALLABLE, orch_args, block_dim=CFG["block_dim"], aicpu_thread_num=CFG["aicpu_thread_num"])
    print(f"[{time.time():.0f}] Worker run OK", flush=True)
    w.close()

    expected = (2.0 + 3.0 + 1) * (2.0 + 3.0 + 2) + (2.0 + 3.0)  # = 47.0
    assert abs(f[0].item() - expected) < 0.01, f"Wrong: f[0]={f[0].item()}"
    print(f"f[0]={f[0].item():.1f} (expected {expected:.1f}) → PASSED", flush=True)


# ---------------------------------------------------------------------------
# Case 2: Worker(level=3) — ChipTask → SubTask (TensorMap dependency)
# ---------------------------------------------------------------------------


def test_case2():
    print("\n" + "=" * 50, flush=True)
    print("Case 2: Worker(level=3) — ChipTask→SubTask (dep)", flush=True)
    print("=" * 50, flush=True)

    a, b, f, orch_args = make_tensors()
    SIZE = f.numel()

    # Shared result (cross-fork via SharedMemory)
    result_shm = SharedMemory(create=True, size=8)
    result_buf = result_shm.buf
    assert result_buf is not None
    struct.pack_into("d", result_buf, 0, -999.0)  # sentinel

    def sub_fn():
        """SubWorker callable: reads f[0] written by ChipTask → stores in shm.
        Uses ctypes (not f[0].item()) to avoid PyTorch re-init in forked child.
        """
        import ctypes  # noqa: PLC0415  # deferred: avoid PyTorch re-init in forked child

        ptr = ctypes.cast(f.data_ptr(), ctypes.POINTER(ctypes.c_float))
        val = float(ptr[0])
        struct.pack_into("d", result_buf, 0, val)

    # Capture pointers BEFORE fork (will be valid in child because they're
    # in the same process address space as the fork)
    chip_callable_ptr = CHIP_CALLABLE.buffer_ptr()  # call method, not property
    orch_args_ptr = orch_args.__ptr__()

    w = Worker(level=3, device_ids=[0], num_sub_workers=1, platform=PLATFORM, runtime=RUNTIME)
    sub_cid = w.register(sub_fn)  # register before fork
    w.init()  # fork → create ChipWorker → start Scheduler
    print(f"[{time.time():.0f}] Worker(level=3) init OK", flush=True)

    def my_orch(w, _args):
        # --- ChipTask: compute f = 47.0 ---
        chip_p = WorkerPayload()
        chip_p.worker_type = WorkerType.CHIP
        chip_p.callable = chip_callable_ptr
        chip_p.args = orch_args_ptr
        chip_p.block_dim = CFG["block_dim"]
        chip_p.aicpu_thread_num = CFG["aicpu_thread_num"]

        chip_result = w.submit(
            WorkerType.CHIP,
            chip_p,
            inputs=[],
            outputs=[SIZE * 4],  # allocate output slot → key for TensorMap
        )
        chip_out_ptr = chip_result.outputs[0].ptr  # key used for dependency inference

        # --- SubWorkerTask: depends on ChipTask via TensorMap ---
        sub_p = WorkerPayload()
        sub_p.worker_type = WorkerType.SUB
        sub_p.callable_id = sub_cid
        w.submit(
            WorkerType.SUB,
            sub_p,
            inputs=[chip_out_ptr],  # TensorMap: ChipTask is producer → fanin
            outputs=[],
        )

    w.run(Task(orch=my_orch, args=None))  # blocks until both tasks consumed
    print(f"[{time.time():.0f}] Worker run OK", flush=True)
    w.close()

    result_val = struct.unpack_from("d", result_buf, 0)[0]
    result_shm.close()
    result_shm.unlink()

    print(f"ChipTask → f[0]={f[0].item():.1f}", flush=True)
    print(f"SubTask read f[0]={result_val:.1f}", flush=True)

    assert abs(f[0].item() - 47.0) < 0.01, f"ChipTask wrong: f[0]={f[0].item()}"
    assert result_val != -999.0, "SubTask never ran"
    assert abs(result_val - 47.0) < 0.01, f"SubTask saw wrong value: {result_val}"
    print("PASSED", flush=True)


if __name__ == "__main__":
    test_case1()
    test_case2()
    print("\n*** ALL TESTS PASSED ***")
