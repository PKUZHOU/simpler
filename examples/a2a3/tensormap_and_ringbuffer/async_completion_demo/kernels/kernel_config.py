"""
Async Completion Demo - Kernel and Orchestration Configuration

Two hardware cards use the existing deferred-completion producer API to
demonstrate a real 2P TGET_ASYNC remote read. The legacy single-card / sim
path stays available for local debugging.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_REPO_ROOT = _KERNELS_ROOT.resolve().parents[4]
_LOCAL_PTO_ISA_INCLUDE = _REPO_ROOT / "3rd" / "pto-isa" / "include"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "async_demo_orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_producer.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_consumer.cpp"), "core_type": "aiv"},
]

if _platform == "a2a3":
    KERNELS.append(
        {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_producer_async.cpp"), "core_type": "aiv"},
    )

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 3,
    "rounds": 1,
}

if _platform == "a2a3":
    RUNTIME_ENV = {
        "PTO2_ENABLE_SDMA": "1",
    }

    DISTRIBUTED_CONFIG = {
        "nranks": 2,
        "root": 0,
        "pto_isa_root": str(_REPO_ROOT / "3rd" / "pto-isa"),
        "win_sync_prefix": 256,
        "buffers": [
            {"name": "in", "dtype": "float32", "count": 128 * 128, "placement": "window"},
            {"name": "out", "dtype": "float32", "count": 128 * 128, "placement": "window"},
            {"name": "result", "dtype": "float32", "count": 128 * 128, "placement": "device"},
        ],
        "inputs": ["in"],
        "outputs": ["out", "result"],
        "args": ["in", "out", "result", "deviceCtx"],
    }

    if _LOCAL_PTO_ISA_INCLUDE.is_dir():
        DISTRIBUTED_CONFIG["comm_include_dirs"] = [str(_LOCAL_PTO_ISA_INCLUDE)]
