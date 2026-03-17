"""
Distributed TREDUCE kernel configuration — aicpu_build_graph runtime.

The AICPU orchestration plugin reads args from runtime->orch_args[],
builds the task graph via the aicpu_build_api, and publishes tasks for
the AICPU scheduler threads.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "treduce_orch.cpp"),
    "function_name": "build_treduce_graph",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "treduce_kernel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 4,
}

RUNTIME_ENV = {
    "PTO_AICPU_BUILD_GRAPH_BUILD_MODE": "1",
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
