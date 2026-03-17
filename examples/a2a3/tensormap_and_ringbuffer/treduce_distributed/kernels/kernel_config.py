"""
Distributed TREDUCE kernel configuration — tensormap_and_ringbuffer runtime.

Device-side orchestration via PTO2Runtime API. The orchestration function
wraps each arg as a PTOParam (tensor or scalar) and submits a single AIV task.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "treduce_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "treduce_kernel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
    "orch_thread_num": 1,
    "rounds": 1,
}

DISTRIBUTED_CONFIG = {
    "nranks": 8,
    "root": 0,
    "comm_include_dirs": ["tests/npu/a2a3/comm/st/testcase"],
    "win_sync_prefix": 256,
    "buffers": [
        {"name": "input",  "dtype": "float32", "count": 256, "placement": "window"},
        {"name": "output", "dtype": "float32", "count": 256, "placement": "device"},
    ],
    "inputs": ["input"],
    "outputs": ["output"],
    "args": ["input", "output", "nranks", "root", "deviceCtx"],
}
