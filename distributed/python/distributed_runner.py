"""
DistributedRunner — compile artifacts and launch N distributed_worker processes.

Usage:
    from distributed_runner import DistributedRunner

    runner = DistributedRunner(
        kernels_dir="examples/a2a3/.../treduce_distributed/kernels",
        platform="a2a3",
        nranks=8,
    )
    runner.compile()
    runner.run()
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SIMPLER_ROOT = Path(__file__).resolve().parent.parent.parent
DISTRIBUTED_ROOT = SIMPLER_ROOT / "distributed"


class DistributedRunner:

    def __init__(
        self,
        kernels_dir: str,
        platform: str = "a2a3",
        nranks: int = 8,
        root: int = 0,
        build_dir: str = None,
        artifact_dir: str = None,
        orch_func: str = None,
    ):
        self.kernels_dir = Path(kernels_dir).resolve()
        self.platform = platform
        self.nranks = nranks
        self.root = root
        self.build_dir = Path(build_dir).resolve() if build_dir else \
            DISTRIBUTED_ROOT / "build" / "cache"
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else \
            DISTRIBUTED_ROOT / "build" / "artifacts"

        self._load_kernel_config()
        self.orch_func = orch_func or self.kernel_config_module.ORCHESTRATION["function_name"]

    def _load_kernel_config(self):
        """Load kernel_config.py from the kernels directory."""
        import importlib.util
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"kernel_config.py not found in {self.kernels_dir}")

        spec = importlib.util.spec_from_file_location("kernel_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.kernel_config_module = module

    def compile(self):
        """Compile all artifacts using simpler's build infrastructure."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        python_dir = SIMPLER_ROOT / "python"
        scripts_dir = SIMPLER_ROOT / "examples" / "scripts"
        sys.path.insert(0, str(python_dir))
        sys.path.insert(0, str(scripts_dir))

        from runtime_builder import RuntimeBuilder
        from kernel_compiler import KernelCompiler
        from elf_parser import extract_text_section

        kcfg = self.kernel_config_module
        runtime_name = kcfg.RUNTIME_CONFIG.get("runtime", "host_build_graph")

        builder = RuntimeBuilder(platform=self.platform)
        kernel_compiler = builder.get_kernel_compiler()

        # Phase 1: Build runtime binaries (host, aicpu, aicore)
        logger.info("=== Phase 1: Building runtime ===")
        host_binary, aicpu_binary, aicore_binary = builder.build(
            runtime_name, str(self.build_dir))

        # Phase 2: Compile orchestration SO
        logger.info("=== Phase 2: Compiling orchestration ===")
        orch_source = kcfg.ORCHESTRATION["source"]
        if not os.path.isabs(orch_source):
            orch_source = str(self.kernels_dir / orch_source)
        orch_binary = kernel_compiler.compile_orchestration(
            runtime_name, orch_source, build_dir=str(self.build_dir))

        # Phase 3: Compile kernels
        logger.info("=== Phase 3: Compiling kernels ===")
        pto_isa_root = str(scripts_dir / "_deps" / "pto-isa")
        arch = "a2a3" if self.platform in ("a2a3", "a2a3sim") else "a5"
        runtime_include_dirs = [
            str(SIMPLER_ROOT / "src" / arch / "runtime" / runtime_name / "runtime")
        ]

        dist_config = getattr(kcfg, "DISTRIBUTED_CONFIG", {})
        extra_includes = runtime_include_dirs + [
            str(DISTRIBUTED_ROOT / "include"),
        ]
        for d in dist_config.get("comm_include_dirs", []):
            p = Path(pto_isa_root) / d if not os.path.isabs(d) else Path(d)
            extra_includes.append(str(p))

        kernel_bins = {}
        for k in kcfg.KERNELS:
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
                kernel_bins[k["func_id"]] = incore_o
            else:
                kernel_bins[k["func_id"]] = extract_text_section(incore_o)

        # Phase 4: Save artifacts
        logger.info("=== Phase 4: Saving artifacts ===")

        def save(name, data):
            path = self.artifact_dir / name
            path.write_bytes(data)
            logger.info(f"  {name}: {len(data)} bytes")

        save("libhost_runtime.so", host_binary)
        save("libaicpu_kernel.so", aicpu_binary)
        save("aicore_kernel.o", aicore_binary)
        save("treduce_orch.so", orch_binary)
        for func_id, data in kernel_bins.items():
            save("treduce_kernel.bin", data)

        logger.info(f"All artifacts saved to {self.artifact_dir}")

    def build_worker(self):
        """Build the distributed_worker C++ executable."""
        worker_build = DISTRIBUTED_ROOT / "build" / "worker"
        worker_build.mkdir(parents=True, exist_ok=True)

        logger.info("=== Building distributed_worker ===")
        cmake_cmd = [
            "cmake", str(DISTRIBUTED_ROOT),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        subprocess.run(cmake_cmd, cwd=str(worker_build), check=True,
                       capture_output=True, text=True)

        make_cmd = ["make", "-j" + str(os.cpu_count() or 4), "distributed_worker"]
        result = subprocess.run(make_cmd, cwd=str(worker_build), check=True,
                                capture_output=True, text=True)
        logger.info("distributed_worker built successfully")

        self.worker_bin = worker_build / "distributed_worker"
        if not self.worker_bin.exists():
            raise RuntimeError(f"Worker binary not found at {self.worker_bin}")

    def run(self):
        """Launch N distributed_worker processes and wait for completion."""
        if not hasattr(self, "worker_bin") or not self.worker_bin.exists():
            self.build_worker()

        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        # Clean up stale files
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

            cmd = [
                str(self.worker_bin),
                "--device-id", str(r),
                "--rank", str(r),
                "--nranks", str(self.nranks),
                "--root", str(self.root),
                "--artifact-dir", str(self.artifact_dir),
                "--rootinfo-file", str(rootinfo_file),
                "--orch-func", self.orch_func,
            ]
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
                logger.info(f"Rank {r}: PASSED")

        # Print summary from each rank
        print()
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            lines = log_path.read_text().strip().split("\n")
            print(f"--- RANK {r} (last 5 lines) ---")
            for line in lines[-5:]:
                print(line)

        print()
        if fail_count == 0:
            print(f"=== ALL {self.nranks} RANKS PASSED ===")
        else:
            print(f"=== {fail_count}/{self.nranks} RANKS FAILED ===")

        # Clean up barrier files
        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()

        return fail_count == 0
