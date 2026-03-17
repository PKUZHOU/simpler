"""
BaseRunner — shared infrastructure for CodeRunner and DistributedRunner.

Provides config loading, compilation, output comparison, and environment utilities.
Subclasses override run() with their specific execution model:
- CodeRunner: in-process execution via ctypes bindings
- DistributedRunner: multi-process execution via distributed_worker binary
"""

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_module_from_path(module_path, module_name="mod"):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_project_root():
    """Get the simpler project root directory (parent of python/)."""
    return Path(__file__).resolve().parent.parent


def setup_logging_if_needed():
    """Setup logging if not already configured."""
    if not logging.getLogger().hasHandlers():
        level_str = os.environ.get("PTO_LOG_LEVEL", "info")
        level_map = {
            "error": logging.ERROR,
            "warn": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        logging.basicConfig(
            level=level_map.get(level_str.lower(), logging.INFO),
            format="[%(levelname)s] %(message)s",
            force=True,
        )


def kernel_config_runtime_env(kernel_config_module, kernels_dir) -> Dict[str, str]:
    """
    Extract RUNTIME_ENV from kernel_config.

    Values whose key ends with _DIR/_PATH are resolved relative to kernels_dir
    when they are not absolute paths.
    """
    runtime_env = getattr(kernel_config_module, "RUNTIME_ENV", None)
    if not isinstance(runtime_env, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in runtime_env.items():
        if not isinstance(k, str):
            continue
        s = str(v)
        if (k.endswith("_DIR") or k.endswith("_PATH")) and s:
            p = Path(s)
            if not p.is_absolute():
                s = str((Path(kernels_dir) / p).resolve())
        out[k] = s
    return out


@contextmanager
def temporary_env(env_updates: Dict[str, str]):
    """Temporarily apply env vars for the duration of the context."""
    old = {k: os.environ.get(k) for k in env_updates.keys()}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


class BaseRunner:
    """
    Base class providing shared config loading, compilation, and verification.

    Subclasses must implement run().
    """

    def __init__(
        self,
        kernels_dir: str,
        golden_path: Optional[str] = None,
        platform: str = "a2a3",
        build_dir: Optional[str] = None,
    ):
        setup_logging_if_needed()

        self.kernels_dir = Path(kernels_dir).resolve()
        self.platform = platform
        self.project_root = str(get_project_root())
        self.build_dir = build_dir

        self._setup_import_paths()

        self._kernel_config = self._load_kernel_config()

        self._golden_module = None
        if golden_path:
            gp = Path(golden_path).resolve()
            if gp.exists():
                self._golden_module = load_module_from_path(
                    gp, f"golden_{id(self)}"
                )

        rc = getattr(self._kernel_config, "RUNTIME_CONFIG", {})
        self.runtime_name = rc.get("runtime", "host_build_graph")
        self.aicpu_thread_num = rc.get("aicpu_thread_num", 3)
        self.orch_thread_num = rc.get("orch_thread_num", 1)
        self.block_dim = rc.get("block_dim", 24)

    # ------------------------------------------------------------------
    # Import paths
    # ------------------------------------------------------------------

    def _setup_import_paths(self):
        """Ensure simpler's python/ and examples/scripts/ are on sys.path."""
        root = Path(self.project_root)
        for d in [root / "python", root / "examples" / "scripts"]:
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_kernel_config(self):
        """Load kernel_config.py from kernels directory."""
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(
                f"kernel_config.py not found in {self.kernels_dir}\n"
                f"Expected: {config_path}"
            )
        return load_module_from_path(config_path, f"kernel_config_{id(self)}")

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_artifacts(
        self,
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[List[str]] = None,
    ) -> Tuple[bytes, bytes, bytes, bytes, List[Tuple[int, bytes]]]:
        """
        Compile runtime, orchestration, and kernels.

        Returns:
            (host_binary, aicpu_binary, aicore_binary, orch_binary,
             kernel_binaries: [(func_id, bytes), ...])
        """
        from runtime_builder import RuntimeBuilder
        from elf_parser import extract_text_section

        builder = RuntimeBuilder(platform=self.platform)
        available = builder.list_runtimes()
        if self.runtime_name not in available:
            raise ValueError(
                f"Runtime '{self.runtime_name}' not available for "
                f"'{self.platform}'. Available: {', '.join(available)}"
            )

        kernel_compiler = builder.get_kernel_compiler()

        logger.info(f"Building runtime: {self.runtime_name} ({self.platform})")
        host_bin, aicpu_bin, aicore_bin = builder.build(
            self.runtime_name, self.build_dir
        )

        orch_src = self._kernel_config.ORCHESTRATION["source"]
        if not os.path.isabs(orch_src):
            orch_src = str(self.kernels_dir / orch_src)
        logger.info(f"Compiling orchestration: {orch_src}")
        orch_bin = kernel_compiler.compile_orchestration(
            self.runtime_name, orch_src, build_dir=self.build_dir
        )

        arch = "a2a3" if self.platform in ("a2a3", "a2a3sim") else "a5"
        rt_include = os.path.join(
            self.project_root, "src", arch, "runtime",
            self.runtime_name, "runtime",
        )
        includes = [rt_include] + (extra_include_dirs or [])

        kernel_binaries: List[Tuple[int, bytes]] = []
        for k in self._kernel_config.KERNELS:
            src = k["source"]
            if not os.path.isabs(src):
                src = str(self.kernels_dir / src)
            logger.info(f"Compiling kernel: {src} (func_id={k['func_id']})")
            incore_o = kernel_compiler.compile_incore(
                src,
                core_type=k.get("core_type", "aiv"),
                pto_isa_root=pto_isa_root,
                extra_include_dirs=includes,
                build_dir=self.build_dir,
            )
            if self.platform.endswith("sim"):
                kernel_binaries.append((k["func_id"], incore_o))
            else:
                kernel_binaries.append(
                    (k["func_id"], extract_text_section(incore_o))
                )

        return host_bin, aicpu_bin, aicore_bin, orch_bin, kernel_binaries

    # ------------------------------------------------------------------
    # Runtime environment
    # ------------------------------------------------------------------

    def get_runtime_env(self) -> Dict[str, str]:
        """Return RUNTIME_ENV from kernel_config, with paths resolved."""
        return kernel_config_runtime_env(self._kernel_config, self.kernels_dir)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_arrays(actual, expected, rtol=1e-5, atol=1e-5, name="output"):
        """
        Compare two sequences element-wise.

        Works with lists, numpy arrays, or torch tensors.
        Returns (ok: bool, mismatches: int, total: int).
        """
        mismatches = 0
        total = len(actual)
        for i, (a, e) in enumerate(zip(actual, expected)):
            if abs(float(a) - float(e)) > atol + rtol * abs(float(e)):
                if mismatches < 5:
                    logger.error(f"  {name}[{i}]: got {a}, expected {e}")
                mismatches += 1
        return (mismatches == 0, mismatches, total)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def run(self):
        """Execute the test. Subclasses must implement this."""
        raise NotImplementedError
