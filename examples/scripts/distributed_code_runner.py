"""
DistributedCodeRunner — compile, prepare data, launch workers, and verify
results for distributed (multi-card) PTO kernel tests.

Parallel to CodeRunner, but handles DISTRIBUTED_CONFIG and spawns N
Python worker processes (one per rank) via distributed_worker.py.

Usage:
    runner = DistributedCodeRunner(
        kernels_dir="path/to/distributed_test/kernels",
        golden_path="path/to/distributed_test/golden.py",
        platform="a2a3", nranks=8,
    )
    runner.run()
"""

import importlib.util
import logging
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

SIMPLER_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

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

DTYPE_TORCH = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint32": torch.int32,
    "uint64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int16": torch.int16,
    "uint16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

DTYPE_NUMPY = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "int16": np.int16,
    "uint16": np.uint16,
    "int8": np.int8,
    "uint8": np.uint8,
}


def _load_module(path, name="mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_tensor_like(value):
    return isinstance(value, (torch.Tensor, np.ndarray))


def _value_to_tensor(value, dtype: str, buffer_name: str) -> torch.Tensor:
    target_dtype = DTYPE_TORCH.get(dtype)
    if target_dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype}' for buffer '{buffer_name}'")

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)

    if target_dtype == torch.bfloat16:
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
    else:
        tensor = tensor.to(target_dtype)
    return tensor.contiguous()


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _read_buffer_as_tensor(path: Path, dtype: str, buffer_name: str) -> torch.Tensor:
    raw = path.read_bytes()
    _, elem_sz = DTYPE_FORMAT.get(dtype, ("f", 4))
    if len(raw) % elem_sz != 0:
        raise ValueError(
            f"Buffer '{buffer_name}' file size {len(raw)} is not aligned to element size {elem_sz}"
        )
    count = len(raw) // elem_sz

    if dtype == "bfloat16":
        array = np.frombuffer(raw, dtype=np.uint16, count=count).copy()
        return torch.from_numpy(array).view(torch.bfloat16).clone()

    np_dtype = DTYPE_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype}' for buffer '{buffer_name}'")
    return torch.from_numpy(np.frombuffer(raw, dtype=np_dtype, count=count).copy())


def _tensor_to_list(tensor: torch.Tensor, dtype: str):
    tensor = tensor.detach().cpu().view(-1)
    if dtype == "bfloat16":
        tensor = tensor.float()
    return tensor.tolist()


class DistributedCodeRunner:

    def __init__(
        self,
        kernels_dir: str,
        golden_path: Optional[str] = None,
        platform: str = "a2a3",
        nranks: Optional[int] = None,
        device_ids: Optional[list[int]] = None,
        root: Optional[int] = None,
        build_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        orch_func: Optional[str] = None,
        pto_isa_commit: Optional[str] = None,
        clone_protocol: str = "ssh",
    ):
        self.kernels_dir = Path(kernels_dir).resolve()
        self.platform = platform
        os.environ["PTO_PLATFORM"] = self.platform
        self.build_dir = Path(build_dir).resolve() if build_dir else \
            SIMPLER_ROOT / "build" / "distributed" / "cache"
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else \
            SIMPLER_ROOT / "build" / "distributed" / "artifacts"
        self.pto_isa_commit = pto_isa_commit
        self.clone_protocol = clone_protocol

        self._load_kernel_config()
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})

        self.nranks = nranks if nranks is not None else dist.get("nranks", 8)
        self.root = root if root is not None else dist.get("root", 0)
        self.orch_func = orch_func or self.kcfg.ORCHESTRATION["function_name"]
        if self.nranks <= 0:
            raise ValueError(f"Distributed nranks must be positive, got {self.nranks}")
        if self.root < 0 or self.root >= self.nranks:
            raise ValueError(
                f"Distributed root must be in [0, {self.nranks}), got {self.root}"
            )

        if device_ids is None:
            self.device_ids = list(range(self.nranks))
        else:
            if len(device_ids) != self.nranks:
                raise ValueError(
                    f"Expected {self.nranks} device ids, got {len(device_ids)}: {device_ids}"
                )
            self.device_ids = list(device_ids)

        self.golden_path = Path(golden_path).resolve() if golden_path else None
        self.golden_mod = None

    def _load_kernel_config(self):
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"kernel_config.py not found in {self.kernels_dir}")
        self.kcfg = _load_module(config_path, "kernel_config")

    def _load_golden(self):
        if self.golden_mod is None and self.golden_path and self.golden_path.exists():
            self.golden_mod = _load_module(self.golden_path, "golden")
        return self.golden_mod

    def _orch_artifact_name(self):
        src = Path(self.kcfg.ORCHESTRATION["source"])
        return src.stem + ".so"

    def _kernel_artifact_name(self, kernel_cfg):
        src = Path(kernel_cfg["source"])
        return src.stem + ".bin"

    def _get_buffer_config(self, name: str):
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        for buf_cfg in dist.get("buffers", []):
            if buf_cfg["name"] == name:
                return buf_cfg
        raise ValueError(
            f"Buffer '{name}' from golden.py not found in DISTRIBUTED_CONFIG['buffers']"
        )

    def _get_dtype_format(self, dtype: str, buffer_name: str):
        fmt = DTYPE_FORMAT.get(dtype)
        if fmt is None:
            raise ValueError(
                f"Unsupported dtype '{dtype}' for buffer '{buffer_name}'"
            )
        return fmt

    # ------------------------------------------------------------------
    # compile()
    # ------------------------------------------------------------------

    def compile(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("aicore", "aicpu", "host"):
            p = self.build_dir / sub
            if p.exists():
                shutil.rmtree(p)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        python_dir = SIMPLER_ROOT / "python"
        sys.path.insert(0, str(python_dir))
        sys.path.insert(0, str(SCRIPTS_DIR))

        from runtime_builder import RuntimeBuilder
        from elf_parser import extract_text_section
        from code_runner import _ensure_pto_isa_root
        from kernel_compiler import KernelCompiler

        pto_isa_root = _ensure_pto_isa_root(
            verbose=True, commit=self.pto_isa_commit,
            clone_protocol=self.clone_protocol)
        if pto_isa_root is None:
            raise EnvironmentError("PTO_ISA_ROOT could not be resolved.")

        dist_config = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        configured_pto_isa_root = dist_config.get("pto_isa_root")
        if configured_pto_isa_root:
            configured_pto_isa_root = Path(configured_pto_isa_root)
            if not configured_pto_isa_root.is_absolute():
                configured_pto_isa_root = (SIMPLER_ROOT / configured_pto_isa_root).resolve()
            if configured_pto_isa_root.is_dir():
                pto_isa_root = str(configured_pto_isa_root)
                logger.info(f"Using configured PTO-ISA root: {pto_isa_root}")

        runtime_name = self.kcfg.RUNTIME_CONFIG.get("runtime", "host_build_graph")
        builder = RuntimeBuilder(platform=self.platform)
        kernel_compiler = KernelCompiler(platform=self.platform)

        logger.info("=== Phase 1: Building runtime ===")
        runtime_binaries = builder.get_binaries(runtime_name, build=True)
        host_binary = runtime_binaries.host_path.read_bytes()
        aicpu_binary = runtime_binaries.aicpu_path.read_bytes()
        aicore_binary = runtime_binaries.aicore_path.read_bytes()

        logger.info("=== Phase 2: Compiling orchestration ===")
        orch_source = self.kcfg.ORCHESTRATION["source"]
        if not os.path.isabs(orch_source):
            orch_source = str(self.kernels_dir / orch_source)
        orch_binary = kernel_compiler.compile_orchestration(
            runtime_name, orch_source, build_dir=str(self.build_dir))

        logger.info("=== Phase 3: Compiling kernels ===")
        if self.platform in ("a2a3", "a2a3sim"):
            arch = "a2a3"
        elif self.platform in ("a5", "a5sim"):
            arch = "a5"
        else:
            arch = "a2a3"

        runtime_base_dir = SIMPLER_ROOT / "src" / arch / "runtime" / runtime_name
        build_config_path = runtime_base_dir / "build_config.py"
        runtime_include_dirs = []
        if build_config_path.is_file():
            spec = importlib.util.spec_from_file_location("build_config", build_config_path)
            assert spec is not None and spec.loader is not None
            bc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bc_module)
            aicore_cfg = bc_module.BUILD_CONFIG.get("aicore", {})
            for path in aicore_cfg.get("include_dirs", []):
                runtime_include_dirs.append(str(runtime_base_dir / path))
        else:
            runtime_include_dirs.append(str(runtime_base_dir / "runtime"))
        runtime_include_dirs.append(str(SIMPLER_ROOT / "src" / "common" / "task_interface"))

        extra_includes = list(runtime_include_dirs) + [
            str(SIMPLER_ROOT / "src" / arch / "platform" / "include"),
        ]
        for d in dist_config.get("comm_include_dirs", []):
            p = Path(pto_isa_root) / d if not os.path.isabs(d) else Path(d)
            extra_includes.append(str(p))

        kernel_bins = {}
        for k in self.kcfg.KERNELS:
            src = k["source"]
            if not os.path.isabs(src):
                src = str(self.kernels_dir / src)
            incore_o = kernel_compiler.compile_incore(
                src,
                core_type=k.get("core_type", "aiv"),
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_includes,
                build_dir=str(self.build_dir),
            )
            if self.platform.endswith("sim"):
                kernel_bins[k["func_id"]] = (k, incore_o)
            else:
                kernel_bins[k["func_id"]] = (k, extract_text_section(incore_o))

        logger.info("=== Phase 4: Saving artifacts ===")

        def save(name, data):
            path = self.artifact_dir / name
            path.write_bytes(data)
            logger.info(f"  {name}: {len(data)} bytes")

        save("libhost_runtime.so", host_binary)
        save("libaicpu_kernel.so", aicpu_binary)
        save("aicore_kernel.o", aicore_binary)
        save(self._orch_artifact_name(), orch_binary)
        for func_id, (kcfg, data) in kernel_bins.items():
            save(self._kernel_artifact_name(kcfg), data)

        logger.info(f"All artifacts saved to {self.artifact_dir}")

    # ------------------------------------------------------------------
    # prepare_data()
    # ------------------------------------------------------------------

    def prepare_data(self):
        golden = self._load_golden()
        if not golden or not hasattr(golden, "generate_distributed_inputs"):
            logger.info("No golden.py or generate_distributed_inputs — skipping data prep")
            return

        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        input_names = set(dist.get("inputs", []))

        for r in range(self.nranks):
            rank_dir = self.artifact_dir / f"rank_{r}"
            rank_dir.mkdir(parents=True, exist_ok=True)

            inputs = golden.generate_distributed_inputs(r, self.nranks, self.root)
            for name, data in inputs:
                if name not in input_names:
                    continue
                buf_cfg = self._get_buffer_config(name)
                if _is_tensor_like(data):
                    tensor = _value_to_tensor(data, buf_cfg["dtype"], name)
                    bin_data = _tensor_to_bytes(tensor)
                elif isinstance(data, (list, tuple)):
                    fmt_char, _ = self._get_dtype_format(buf_cfg["dtype"], name)
                    bin_data = struct.pack(f"<{len(data)}{fmt_char}", *data)
                else:
                    raise TypeError(
                        f"Unsupported distributed input type for '{name}': {type(data)}"
                    )
                path = rank_dir / f"{name}.bin"
                path.write_bytes(bin_data)
                logger.debug(f"  rank_{r}/{name}.bin: {len(bin_data)} bytes")

        logger.info(f"Prepared data for {self.nranks} ranks in {self.artifact_dir}")

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def _build_worker_cmd(self, r):
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "distributed_worker.py"),
            "--device-id", str(self.device_ids[r]),
            "--rank", str(r),
            "--nranks", str(self.nranks),
            "--root", str(self.root),
            "--artifact-dir", str(self.artifact_dir),
            "--rootinfo-file", str(rootinfo_file),
            "--data-dir", str(self.artifact_dir / f"rank_{r}"),
            "--orch-file", self._orch_artifact_name(),
            "--orch-func", self.orch_func,
        ]

        phase2_cfg = dist.get("phase2", {})
        phase2_orch_func = phase2_cfg.get("orch_func")
        if phase2_orch_func:
            cmd += ["--phase2-orch-func", phase2_orch_func]

        rt_cfg = getattr(self.kcfg, "RUNTIME_CONFIG", {})
        cmd += ["--aicpu-thread-num", str(rt_cfg.get("aicpu_thread_num", 1))]
        cmd += ["--block-dim", str(rt_cfg.get("block_dim", 1))]
        cmd += ["--orch-thread-num", str(rt_cfg.get("orch_thread_num", 0))]

        win_sync = dist.get("win_sync_prefix", 0)
        if win_sync:
            cmd += ["--win-sync-prefix", str(win_sync)]

        for buf in dist.get("buffers", []):
            spec = (
                f"{buf['name']}:{buf['dtype']}:{buf['count']}:"
                f"{int(buf.get('data_prefix_elems', 0) or 0)}"
            )
            if buf["placement"] == "window":
                cmd += ["--win-buffer", spec]
            else:
                cmd += ["--dev-buffer", spec]

        for name in dist.get("inputs", []):
            cmd += ["--load", name]

        for name in dist.get("outputs", []):
            cmd += ["--save", name]

        for tok in dist.get("args", []):
            cmd += ["--arg", tok]

        for k in self.kcfg.KERNELS:
            cmd += ["--kernel-bin",
                     f"{k['func_id']}:{self._kernel_artifact_name(k)}"]

        return cmd

    def run(self):
        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()
        if rootinfo_file.exists():
            rootinfo_file.unlink()

        shm_dir = Path("/dev/shm")
        if shm_dir.is_dir():
            for f in shm_dir.glob("simpler_comm_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

        logger.info(f"=== Launching {self.nranks} workers ===")

        procs = []
        log_files = []
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            log_f = open(log_path, "w")
            log_files.append(log_f)

            cmd = self._build_worker_cmd(r)
            env = os.environ.copy()
            runtime_env = getattr(self.kcfg, "RUNTIME_ENV", None)
            if isinstance(runtime_env, dict):
                env.update(runtime_env)

            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            procs.append(proc)

        fail_count = 0
        for r, proc in enumerate(procs):
            proc.wait()
            log_files[r].close()
            if proc.returncode != 0:
                fail_count += 1
                logger.error(f"Rank {r}: FAILED (exit code {proc.returncode})")
            else:
                logger.info(f"Rank {r}: OK")

        print()
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            lines = log_path.read_text().strip().split("\n")
            print(f"--- RANK {r} (last 5 lines) ---")
            for line in lines[-5:]:
                print(line)

        print()
        if fail_count == 0:
            print(f"=== ALL {self.nranks} RANKS COMPLETED ===")
        else:
            print(f"=== {fail_count}/{self.nranks} RANKS FAILED ===")

        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()

        self._run_ok = (fail_count == 0)
        return self._run_ok

    # ------------------------------------------------------------------
    # verify()
    # ------------------------------------------------------------------

    def verify(self):
        golden = self._load_golden()
        if not golden or not hasattr(golden, "compute_golden"):
            logger.info("No golden.py or compute_golden — skipping verification")
            return True

        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        output_names = dist.get("outputs", [])
        buf_map = {b["name"]: b for b in dist.get("buffers", [])}

        rtol = getattr(golden, "RTOL", 1e-5)
        atol = getattr(golden, "ATOL", 1e-5)
        ignore_prefix = int(getattr(golden, "IGNORE_PREFIX_ELEMS", 0) or 0)

        all_ok = True
        for rank in range(self.nranks):
            rank_dir = self.artifact_dir / f"rank_{rank}"
            actual_outputs = {}
            for name in output_names:
                path = rank_dir / f"{name}.bin"
                if not path.exists():
                    logger.error(f"Output file not found: {path}")
                    all_ok = False
                    continue
                dtype = buf_map.get(name, {}).get("dtype", "float32")
                actual_outputs[name] = _read_buffer_as_tensor(path, dtype, name).view(-1)

            if len(actual_outputs) != len(output_names):
                continue

            generated_items = []
            if hasattr(golden, "generate_distributed_inputs"):
                generated_items = list(golden.generate_distributed_inputs(rank, self.nranks, self.root))

            tensor_mode = any(_is_tensor_like(data) for _, data in generated_items)
            params = {"nranks": self.nranks, "root": self.root, "rank": rank}

            if tensor_mode:
                tensors = {}
                for name, data in generated_items:
                    if name not in buf_map:
                        continue
                    tensors[name] = _value_to_tensor(data, buf_map[name]["dtype"], name)
                for name in output_names:
                    if name not in tensors:
                        tensors[name] = torch.zeros_like(actual_outputs[name])
                golden.compute_golden(tensors, params)
                for name in output_names:
                    actual = actual_outputs[name]
                    expected = tensors[name].detach().cpu().contiguous().view(-1)
                    if ignore_prefix > 0:
                        actual = actual[ignore_prefix:]
                        expected = expected[ignore_prefix:]
                    if not torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol):
                        mismatches = torch.nonzero(
                            torch.abs(actual.float() - expected.float()) > atol + rtol * torch.abs(expected.float())
                        ).view(-1)
                        for idx in mismatches[:3].tolist():
                            logger.error(
                                f"  rank {rank} {name}[{idx}]: got {actual[idx].item()}, expected {expected[idx].item()}"
                            )
                        logger.error(
                            f"VERIFY FAILED: rank {rank} {name} — {mismatches.numel()}/{actual.numel()} mismatches"
                        )
                        all_ok = False
                    else:
                        logger.info(f"VERIFY PASSED: rank {rank} {name} — {actual.numel()} elements correct")
                        if rank == 0 and actual.numel() >= 5:
                            logger.info(f"  Sample: {actual[:5].tolist()}")
                continue

            tensors = {}
            for name, data in generated_items:
                if name in output_names:
                    tensors[name] = [0] * actual_outputs[name].numel()
                else:
                    tensors[name] = list(data) if isinstance(data, tuple) else data
            for name in output_names:
                tensors.setdefault(name, [0] * actual_outputs[name].numel())
            golden.compute_golden(tensors, params)

            for name in output_names:
                dtype = buf_map.get(name, {}).get("dtype", "float32")
                actual = _tensor_to_list(actual_outputs[name], dtype)
                expected = tensors[name]
                if ignore_prefix > 0:
                    actual = actual[ignore_prefix:]
                    expected = expected[ignore_prefix:]

                mismatches = 0
                for i, (a, e) in enumerate(zip(actual, expected)):
                    if abs(a - e) > atol + rtol * abs(e):
                        if mismatches < 3:
                            logger.error(f"  rank {rank} {name}[{i}]: got {a}, expected {e}")
                        mismatches += 1
                if mismatches > 0:
                    logger.error(f"VERIFY FAILED: rank {rank} {name} — {mismatches}/{len(actual)} mismatches")
                    all_ok = False
                else:
                    logger.info(f"VERIFY PASSED: rank {rank} {name} — {len(actual)} elements correct")
                    if rank == 0 and len(actual) >= 5:
                        logger.info(f"  Sample: {actual[:5]}")

        if all_ok:
            print("\n=== VERIFICATION PASSED ===\n")
        else:
            print("\n=== VERIFICATION FAILED ===\n")

        return all_ok

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_all(self, skip_compile=False, skip_verify=False):
        if not skip_compile:
            self.compile()

        if self.golden_path:
            self.prepare_data()

        success = self.run()

        if success and self.golden_path and not skip_verify:
            success = self.verify()

        return success
