"""
Distributed TREDUCE kernel configuration.

Multi-card collective reduce (Sum) across N ranks using PTO comm instructions.
Communication addresses are set up by the distributed_worker via HCCL.
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
    "runtime": "host_build_graph",
    "aicpu_thread_num": 1,
    "block_dim": 1,
}

DISTRIBUTED_CONFIG = {
    "nranks": 8,
    "root": 0,
    "comm_include_dirs": ["tests/npu/a2a3/comm/st/testcase"],
}
