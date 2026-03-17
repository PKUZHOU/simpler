"""
DistributedRunner — compile, prepare data, launch workers, and verify results.

Inherits config loading and compilation from BaseRunner.
Adds multi-process worker launching, per-rank data preparation,
and file-based golden verification for distributed (multi-card) kernels.

The multi-card graph is defined by kernel_config.py (DISTRIBUTED_CONFIG) and
golden.py (generate_distributed_inputs / compute_golden).

Usage:
    runner = DistributedRunner(
        kernels_dir="examples/a2a3/.../treduce_distributed/kernels",
        golden_path="examples/a2a3/.../treduce_distributed/golden.py",
        platform="a2a3", nranks=8,
    )
    runner.compile()
    runner.prepare_data()
    runner.build_worker()
    runner.run()
    runner.verify()
"""

import logging
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

from base_runner import BaseRunner, get_project_root

logger = logging.getLogger(__name__)

SIMPLER_ROOT = get_project_root()
DISTRIBUTED_ROOT = SIMPLER_ROOT / "distributed"

DTYPE_FORMAT = {
    "float32": ("f", 4), "float64": ("d", 8),
    "int32": ("i", 4), "int64": ("q", 8),
    "uint32": ("I", 4), "uint64": ("Q", 8),
    "float16": ("e", 2), "int16": ("h", 2), "uint16": ("H", 2),
    "int8": ("b", 1), "uint8": ("B", 1),
}


class DistributedRunner(BaseRunner):

    def __init__(
        self,
        kernels_dir: str,
        golden_path: Optional[str] = None,
        platform: str = "a2a3",
        nranks: Optional[int] = None,
        root: Optional[int] = None,
        build_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        orch_func: Optional[str] = None,
    ):
        super().__init__(kernels_dir, golden_path, platform, build_dir)

        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else \
            DISTRIBUTED_ROOT / "build" / "artifacts"
        if self.build_dir is None:
            self.build_dir = str(DISTRIBUTED_ROOT / "build" / "cache")

        dist = getattr(self._kernel_config, "DISTRIBUTED_CONFIG", {})
        self.nranks = nranks if nranks is not None else dist.get("nranks", 8)
        self.root = root if root is not None else dist.get("root", 0)
        self.orch_func = orch_func or self._kernel_config.ORCHESTRATION["function_name"]

    # ------------------------------------------------------------------
    # Artifact names derived from kernel_config
    # ------------------------------------------------------------------

    def _orch_artifact_name(self):
        src = Path(self._kernel_config.ORCHESTRATION["source"])
        return src.stem + ".so"

    def _kernel_artifact_name(self, kernel_cfg):
        src = Path(kernel_cfg["source"])
        return src.stem + ".bin"

    # ------------------------------------------------------------------
    # compile()
    # ------------------------------------------------------------------

    def compile(self):
        """Compile all artifacts using BaseRunner.compile_artifacts()."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        build_path = Path(self.build_dir)
        for sub in ("aicore", "aicpu", "host"):
            p = build_path / sub
            if p.exists():
                shutil.rmtree(p)
        build_path.mkdir(parents=True, exist_ok=True)

        scripts_dir = SIMPLER_ROOT / "examples" / "scripts"
        pto_isa_root = str(scripts_dir / "_deps" / "pto-isa")

        dist_config = getattr(self._kernel_config, "DISTRIBUTED_CONFIG", {})
        extra_includes = [str(DISTRIBUTED_ROOT / "include")]
        for d in dist_config.get("comm_include_dirs", []):
            p = Path(pto_isa_root) / d if not os.path.isabs(d) else Path(d)
            extra_includes.append(str(p))

        host_bin, aicpu_bin, aicore_bin, orch_bin, kernel_binaries = \
            self.compile_artifacts(
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_includes,
            )

        logger.info("=== Saving artifacts ===")

        def save(name, data):
            path = self.artifact_dir / name
            path.write_bytes(data)
            logger.info(f"  {name}: {len(data)} bytes")

        save("libhost_runtime.so", host_bin)
        save("libaicpu_kernel.so", aicpu_bin)
        save("aicore_kernel.o", aicore_bin)
        save(self._orch_artifact_name(), orch_bin)
        for func_id, data in kernel_binaries:
            for k in self._kernel_config.KERNELS:
                if k["func_id"] == func_id:
                    save(self._kernel_artifact_name(k), data)
                    break

        logger.info(f"All artifacts saved to {self.artifact_dir}")

    # ------------------------------------------------------------------
    # prepare_data()
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Generate per-rank input .bin files from golden.py."""
        golden = self._golden_module
        if not golden or not hasattr(golden, "generate_distributed_inputs"):
            logger.info("No golden.py or generate_distributed_inputs — skipping data prep")
            return

        for r in range(self.nranks):
            rank_dir = self.artifact_dir / f"rank_{r}"
            rank_dir.mkdir(parents=True, exist_ok=True)

            inputs = golden.generate_distributed_inputs(r, self.nranks, self.root)
            for name, data in inputs:
                if isinstance(data, (list, tuple)):
                    dist = getattr(self._kernel_config, "DISTRIBUTED_CONFIG", {})
                    buf_cfg = None
                    for b in dist.get("buffers", []):
                        if b["name"] == name:
                            buf_cfg = b
                            break
                    dtype = buf_cfg["dtype"] if buf_cfg else "float32"
                    fmt_char, _ = DTYPE_FORMAT.get(dtype, ("f", 4))
                    bin_data = struct.pack(f"<{len(data)}{fmt_char}", *data)
                    path = rank_dir / f"{name}.bin"
                    path.write_bytes(bin_data)
                    logger.debug(f"  rank_{r}/{name}.bin: {len(bin_data)} bytes")

        logger.info(f"Prepared data for {self.nranks} ranks in {self.artifact_dir}")

    # ------------------------------------------------------------------
    # build_worker()
    # ------------------------------------------------------------------

    def _worker_platform(self):
        """Map runner platform to worker DISTRIBUTED_PLATFORM."""
        if self.platform.endswith("sim"):
            return "sim"
        return "onboard"

    def build_worker(self):
        """Build the distributed_worker C++ executable."""
        worker_build = DISTRIBUTED_ROOT / "build" / "worker"
        worker_build.mkdir(parents=True, exist_ok=True)

        dist_platform = self._worker_platform()
        logger.info(f"=== Building distributed_worker (platform={dist_platform}) ===")
        cmake_cmd = [
            "cmake", str(DISTRIBUTED_ROOT),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DDISTRIBUTED_PLATFORM={dist_platform}",
        ]
        subprocess.run(cmake_cmd, cwd=str(worker_build), check=True,
                       capture_output=True, text=True)

        make_cmd = ["make", "-j" + str(os.cpu_count() or 4), "distributed_worker"]
        subprocess.run(make_cmd, cwd=str(worker_build), check=True,
                       capture_output=True, text=True)
        logger.info("distributed_worker built successfully")

        self.worker_bin = worker_build / "distributed_worker"
        if not self.worker_bin.exists():
            raise RuntimeError(f"Worker binary not found at {self.worker_bin}")

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def _build_worker_cmd(self, r):
        """Build the CLI command for a single rank from DISTRIBUTED_CONFIG."""
        dist = getattr(self._kernel_config, "DISTRIBUTED_CONFIG", {})
        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        cmd = [
            str(self.worker_bin),
            "--device-id", str(r),
            "--rank", str(r),
            "--nranks", str(self.nranks),
            "--root", str(self.root),
            "--artifact-dir", str(self.artifact_dir),
            "--rootinfo-file", str(rootinfo_file),
            "--data-dir", str(self.artifact_dir / f"rank_{r}"),
            "--orch-file", self._orch_artifact_name(),
            "--orch-func", self.orch_func,
        ]

        cmd += ["--aicpu-thread-num", str(self.aicpu_thread_num)]
        cmd += ["--block-dim", str(self.block_dim)]
        cmd += ["--orch-thread-num", str(self.orch_thread_num)]

        win_sync = dist.get("win_sync_prefix", 0)
        if win_sync:
            cmd += ["--win-sync-prefix", str(win_sync)]

        for buf in dist.get("buffers", []):
            spec = f"{buf['name']}:{buf['dtype']}:{buf['count']}"
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

        for k in self._kernel_config.KERNELS:
            cmd += ["--kernel-bin",
                     f"{k['func_id']}:{self._kernel_artifact_name(k)}"]

        # Pass RUNTIME_ENV as --env KEY=VALUE
        runtime_env = self.get_runtime_env()
        for k, v in runtime_env.items():
            cmd += ["--env", f"{k}={v}"]

        return cmd

    def run(self):
        """Launch N distributed_worker processes and wait for completion."""
        if not hasattr(self, "worker_bin") or not self.worker_bin.exists():
            self.build_worker()

        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()
        if rootinfo_file.exists():
            rootinfo_file.unlink()

        logger.info(f"=== Launching {self.nranks} workers ===")

        procs = []
        log_files = []
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            log_f = open(log_path, "w")
            log_files.append(log_f)

            cmd = self._build_worker_cmd(r)
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
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
        """Read output .bin files from root rank and compare with golden."""
        golden = self._golden_module
        if not golden or not hasattr(golden, "compute_golden"):
            logger.info("No golden.py or compute_golden — skipping verification")
            return True

        dist = getattr(self._kernel_config, "DISTRIBUTED_CONFIG", {})
        output_names = dist.get("outputs", [])
        buf_map = {b["name"]: b for b in dist.get("buffers", [])}

        root_dir = self.artifact_dir / f"rank_{self.root}"
        tensors = {}

        for name in output_names:
            path = root_dir / f"{name}.bin"
            if not path.exists():
                logger.error(f"Output file not found: {path}")
                return False
            raw = path.read_bytes()
            dtype = buf_map.get(name, {}).get("dtype", "float32")
            fmt_char, elem_sz = DTYPE_FORMAT.get(dtype, ("f", 4))
            count = len(raw) // elem_sz
            tensors[name] = list(struct.unpack(f"<{count}{fmt_char}", raw))

        params = {"nranks": self.nranks, "root": self.root}
        golden.compute_golden(tensors, params)

        rtol = getattr(golden, "RTOL", 1e-5)
        atol = getattr(golden, "ATOL", 1e-5)

        all_ok = True
        for name in output_names:
            path = root_dir / f"{name}.bin"
            raw = path.read_bytes()
            dtype = buf_map.get(name, {}).get("dtype", "float32")
            fmt_char, elem_sz = DTYPE_FORMAT.get(dtype, ("f", 4))
            count = len(raw) // elem_sz
            actual = list(struct.unpack(f"<{count}{fmt_char}", raw))
            expected = tensors[name]

            ok, mismatches, total = self.compare_arrays(
                actual, expected, rtol=rtol, atol=atol, name=name)
            if not ok:
                logger.error(f"VERIFY FAILED: {name} — {mismatches}/{total} mismatches")
                all_ok = False
            else:
                logger.info(f"VERIFY PASSED: {name} — {total} elements correct")
                if total >= 5:
                    logger.info(f"  Sample: {actual[:5]}")

        if all_ok:
            print("\n=== VERIFICATION PASSED ===\n")
        else:
            print("\n=== VERIFICATION FAILED ===\n")

        return all_ok
