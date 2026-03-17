"""Backward-compatible redirect — DistributedRunner now lives in python/distributed_runner.py."""

import sys
from pathlib import Path

_python_dir = str(Path(__file__).resolve().parent.parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from distributed_runner import DistributedRunner  # noqa: F401, E402

__all__ = ["DistributedRunner"]
