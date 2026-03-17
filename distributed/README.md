# simpler 分布式扩展 (Distributed Extension)

在昇腾 NPU 多卡环境下，通过 HCCL 建立 RDMA 通信上下文，利用 simpler 的三种构图 runtime 执行 PTO 集合通信 kernel。

## 架构

```
run_distributed_example.py  (CLI 入口)
    │
    ▼
DistributedRunner           (Python 编排层)
    ├── compile()           调用 simpler 的 RuntimeBuilder + KernelCompiler
    ├── prepare_data()      从 golden.py 生成 per-rank 输入 .bin 文件
    ├── build_worker()      cmake + make distributed_worker
    ├── run()               并行启动 N 个 distributed_worker 进程
    └── verify()            对比 root rank 输出与 golden
         │
         ▼  (per-rank process)
distributed_worker          (C++ 通用 worker)
    ├── ACL/HCCL 初始化
    ├── RootInfo 文件交换
    ├── HcclAllocComResourceByTiling → windowsIn[] GVA 地址
    ├── dlopen libhost_runtime.so (simpler)
    ├── set_device → init_runtime → launch_runtime → finalize_runtime
    └── 数据 load/save (来自 CLI --load/--save)
```

## 支持的 Runtime

| Runtime | 编排位置 | `orch_thread_num` | 典型 `block_dim` | 说明 |
|---------|---------|-------------------|-----------------|------|
| `host_build_graph` | Host 侧 | 0 (默认) | 1 | Host 构图，AICPU 调度 |
| `aicpu_build_graph` | AICPU 侧 | 0 | 4 | AICPU 构图 + 调度，需 `PTO_AICPU_BUILD_GRAPH_BUILD_MODE=1` |
| `tensormap_and_ringbuffer` | AICPU 侧 (PTO2) | 1 | 3 | 设备端编排，1 编排器 + 3 调度器，block_dim 须被 3 整除 |

## 与 simpler 现有代码的关系

**零侵入**：`src/`、`python/`、`examples/scripts/` 不做任何修改。

| 层 | simpler 现有 | distributed 新增 | 交互方式 |
|----|-------------|-----------------|---------|
| 编译 | RuntimeBuilder, KernelCompiler | DistributedRunner.compile() | import 复用 |
| C API | libhost_runtime.so | distributed_worker (dlopen) | dlopen + dlsym |
| Kernel | kernel_entry 签名 + PTO 指令 | treduce_kernel.cpp | 完全兼容 |
| 示例约定 | kernel_config.py + golden.py | 扩展 DISTRIBUTED_CONFIG + RUNTIME_CONFIG | 向后兼容 |

## 目录结构

```
distributed/
├── README.md                 本文件
├── CMakeLists.txt            构建 distributed_worker
├── include/
│   └── hccl_context.h        HcclDeviceContext 结构定义
├── src/
│   └── distributed_worker.cpp  per-device C++ 通用 worker
└── python/
    ├── distributed_runner.py   DistributedRunner 类
    └── run_distributed_example.py  CLI 入口

examples/a2a3/<runtime>/treduce_distributed/
├── golden.py                 分布式 golden (generate + verify)
└── kernels/
    ├── kernel_config.py      RUNTIME_CONFIG + DISTRIBUTED_CONFIG + ORCHESTRATION + KERNELS
    ├── orchestration/
    │   └── treduce_orch.cpp  编排函数 (不同 runtime 签名不同)
    └── aiv/
        └── treduce_kernel.cpp  PTO TREDUCE kernel (三种 runtime 共用)
```

已验证的三组示例：
- `examples/a2a3/host_build_graph/treduce_distributed/`
- `examples/a2a3/aicpu_build_graph/treduce_distributed/`
- `examples/a2a3/tensormap_and_ringbuffer/treduce_distributed/`

## 快速开始

### 前置条件

- 昇腾 910B / A2 硬件，8 卡环境
- CANN 8.x 已安装并 source 过 `set_env.sh`
- Python 3.8+

### 一键运行（推荐）

```bash
cd simpler/
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# host_build_graph runtime
python distributed/python/run_distributed_example.py \
  -k examples/a2a3/host_build_graph/treduce_distributed/kernels \
  -p a2a3 --nranks 8

# aicpu_build_graph runtime
python distributed/python/run_distributed_example.py \
  -k examples/a2a3/aicpu_build_graph/treduce_distributed/kernels \
  -p a2a3 --nranks 8

# tensormap_and_ringbuffer runtime
python distributed/python/run_distributed_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/treduce_distributed/kernels \
  -p a2a3 --nranks 8
```

该命令会自动：
1. 编译 runtime + 编排 + kernel 产物
2. 生成 per-rank 输入数据
3. 构建 `distributed_worker` 可执行文件
4. 启动 8 个 worker 进程
5. root rank 验证结果并输出 PASS/FAIL

### 分步运行

```bash
# 1. 编译产物
python distributed/python/run_distributed_example.py \
  -k examples/a2a3/host_build_graph/treduce_distributed/kernels \
  -p a2a3 --nranks 8 --skip-compile

# 2. 手动构建 worker
cd distributed/build/worker
cmake ../.. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. 手动启动单个 rank（调试用）
./distributed_worker \
  --device-id 0 --rank 0 --nranks 8 --root 0 \
  --artifact-dir ../artifacts \
  --rootinfo-file ../artifacts/rootinfo.bin \
  --data-dir ../artifacts/rank_0 \
  --orch-file treduce_orch.so --orch-func build_treduce_graph \
  --win-buffer input:float32:256 --dev-buffer output:float32:256 \
  --load input --save output \
  --arg input --arg output --arg nranks --arg root --arg deviceCtx \
  --kernel-bin 0:treduce_kernel.bin \
  --aicpu-thread-num 1 --block-dim 1 --orch-thread-num 0
```

### CLI 参数 (run_distributed_example.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-k` / `--kernels` | (必需) | kernel_config.py 所在目录 |
| `-p` / `--platform` | a2a3 | 目标平台 |
| `--nranks` | 8 | 卡数 |
| `--root` | 0 | 集合通信的 root rank |
| `--orch-func` | (从 kernel_config) | 编排函数名 |
| `--skip-compile` | false | 跳过编译步骤 |
| `--skip-build-worker` | false | 跳过 worker 构建 |
| `-v` | false | 详细输出 |

### Worker CLI 参数 (distributed_worker)

| 参数 | 说明 |
|------|------|
| `--device-id N` | NPU 设备 ID |
| `--rank R` | 本进程的 rank 编号 |
| `--nranks N` | 总卡数 |
| `--root R` | root rank |
| `--artifact-dir PATH` | 编译产物目录 |
| `--rootinfo-file PATH` | HCCL RootInfo 交换文件 |
| `--data-dir PATH` | per-rank 数据目录 |
| `--orch-file FILE` | 编排 SO 文件名 |
| `--orch-func NAME` | 编排函数名 |
| `--win-buffer name:dtype:count` | HCCL RDMA window 缓冲区 |
| `--dev-buffer name:dtype:count` | 普通 device 缓冲区 |
| `--load name` | 从 data-dir 加载输入 |
| `--save name` | 完成后保存输出到 data-dir |
| `--arg token` | 传给 init_runtime 的参数 |
| `--kernel-bin id:file` | kernel binary 映射 |
| `--aicpu-thread-num N` | AICPU 线程数 (默认 1) |
| `--block-dim N` | AICore block 维度 (默认 1) |
| `--orch-thread-num N` | 编排线程数 (默认 0，tensormap_and_ringbuffer 需设 1) |
| `--win-sync-prefix bytes` | HCCL window 同步前缀字节数 |

## 添加新的分布式 kernel

1. 选择 runtime，在 `examples/a2a3/<runtime>/` 下新建目录
2. 创建 `kernels/kernel_config.py`，包含 `RUNTIME_CONFIG`、`DISTRIBUTED_CONFIG`、`ORCHESTRATION`、`KERNELS`
3. 编写 orchestration（不同 runtime 签名不同，参考已有示例）
4. 编写 kernel（使用 `kernel_entry` 签名）
5. 编写 `golden.py`（提供 `generate_distributed_inputs` 和 `compute_golden`）
6. 运行：`python distributed/python/run_distributed_example.py -k <your_kernels_dir>`

## kernel_config.py 配置格式

```python
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",   # 或 "aicpu_build_graph" / "tensormap_and_ringbuffer"
    "aicpu_thread_num": 1,           # AICPU 线程数
    "block_dim": 1,                  # AICore block 维度
    "orch_thread_num": 0,            # 编排线程数 (tensormap_and_ringbuffer 需设 1)
}

DISTRIBUTED_CONFIG = {
    "nranks": 8,
    "root": 0,
    "comm_include_dirs": [           # 编译 kernel 时额外 include 路径 (相对于 pto-isa root)
        "tests/npu/a2a3/comm/st/testcase",
    ],
    "win_sync_prefix": 256,          # HCCL window 前缀预留字节
    "buffers": [
        {"name": "input",  "dtype": "float32", "count": 256, "placement": "window"},
        {"name": "output", "dtype": "float32", "count": 256, "placement": "device"},
    ],
    "inputs": ["input"],             # 需要从 golden 加载的 buffer
    "outputs": ["output"],           # 需要保存验证的 buffer
    "args": ["input", "output", "nranks", "root", "deviceCtx"],  # 传给编排的参数
}

ORCHESTRATION = {
    "source": "orchestration/treduce_orch.cpp",
    "function_name": "build_treduce_graph",  # 编排函数名
}

KERNELS = [
    {"func_id": 0, "source": "aiv/treduce_kernel.cpp", "core_type": "aiv"},
]
```

## 测试结果

三种 runtime 均通过 8 卡 TREDUCE 验证 (256 float 元素，reduce sum)：

```
=== ALL 8 RANKS COMPLETED ===
VERIFY PASSED: output — 256 elements correct
  Sample: [2800.0, 2808.0, 2816.0, 2824.0, 2832.0]
=== VERIFICATION PASSED ===
```
