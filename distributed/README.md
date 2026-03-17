# simpler 分布式扩展 (Distributed Extension)

在昇腾 NPU 多卡环境下，通过 HCCL 建立 RDMA 通信上下文，利用 simpler 的 AICPU 编排执行 PTO 集合通信 kernel。

## 架构

```
run_distributed_example.py  (CLI 入口)
    │
    ▼
DistributedRunner           (Python 编排层)
    ├── compile()           调用 simpler 的 RuntimeBuilder + KernelCompiler
    ├── build_worker()      cmake + make distributed_worker
    └── run()               并行启动 N 个 distributed_worker 进程
         │
         ▼  (per-rank process)
distributed_worker          (C++ worker)
    ├── ACL/HCCL 初始化
    ├── RootInfo 文件交换
    ├── HcclAllocComResourceByTiling → windowsIn[] GVA 地址
    ├── dlopen libhost_runtime.so (simpler)
    ├── set_device → init_runtime → launch_runtime → finalize_runtime
    └── root rank 验证结果
```


## 与 simpler 现有代码的关系

**零侵入**：`src/`、`python/`、`examples/scripts/` 不做任何修改。

| 层 | simpler 现有 | distributed 新增 | 交互方式 |
|----|-------------|-----------------|---------|
| 编译 | RuntimeBuilder, KernelCompiler | DistributedRunner.compile() | import 复用 |
| C API | libhost_runtime.so | distributed_worker (dlopen) | dlopen + dlsym |
| Kernel | kernel_entry 签名 + PTO 指令 | treduce_kernel.cpp | 完全兼容 |
| 示例约定 | kernel_config.py + golden.py | 扩展 DISTRIBUTED_CONFIG | 向后兼容 |

## 目录结构

```
distributed/
├── README.md                 本文件
├── CMakeLists.txt            构建 distributed_worker
├── include/
│   └── hccl_context.h        HcclDeviceContext 结构定义
├── src/
│   └── distributed_worker.cpp  per-device C++ worker
└── python/
    ├── distributed_runner.py   DistributedRunner 类
    └── run_distributed_example.py  CLI 入口

examples/a2a3/host_build_graph/treduce_distributed/
├── golden.py                 分布式 golden
└── kernels/
    ├── kernel_config.py      kernel + 编排 + 分布式配置
    ├── orchestration/
    │   └── treduce_orch.cpp  编排函数
    └── aiv/
        └── treduce_kernel.cpp  PTO TREDUCE kernel
```

## 快速开始

### 前置条件

- 昇腾 910B / A2 硬件，8 卡环境
- CANN 8.x 已安装并 source 过 `set_env.sh`
- Python 3.8+

### 一键运行（推荐）

```bash
cd simpler/
source /usr/local/Ascend/cann-8.5.0/set_env.sh

python distributed/python/run_distributed_example.py \
  -k examples/a2a3/host_build_graph/treduce_distributed/kernels \
  -p a2a3 --nranks 8
```

该命令会自动：
1. 编译 runtime + 编排 + kernel 产物
2. 构建 `distributed_worker` 可执行文件
3. 启动 8 个 worker 进程
4. root rank 验证结果并输出 PASS/FAIL

### 分步运行

```bash
# 1. 编译产物
python distributed/python/run_distributed_example.py \
  -k examples/a2a3/host_build_graph/treduce_distributed/kernels \
  -p a2a3 --nranks 8 --skip-compile  # 跳过（如果已编译）

# 2. 手动构建 worker
cd distributed/build/worker
cmake ../.. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. 手动启动单个 rank（调试用）
./distributed_worker \
  --device-id 0 --rank 0 --nranks 8 --root 0 \
  --artifact-dir ../artifacts \
  --rootinfo-file ../artifacts/rootinfo.bin \
  --orch-func build_treduce_graph
```

### CLI 参数

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

## 添加新的分布式 kernel

1. 在 `examples/a2a3/host_build_graph/` 下新建目录
2. 创建 `kernels/kernel_config.py`，包含 `DISTRIBUTED_CONFIG`
3. 编写 orchestration 和 kernel（使用 `kernel_entry` 签名）
4. 编写 `golden.py`（提供 `generate_distributed_inputs` 和 `compute_golden`）
5. 运行：`python distributed/python/run_distributed_example.py -k <your_kernels_dir>`

## kernel_config.py 分布式扩展字段

```python
DISTRIBUTED_CONFIG = {
    "nranks": 8,                 # 默认卡数
    "root": 0,                   # 默认 root rank
    "comm_include_dirs": [       # 编译 kernel 时额外的 include 路径
        "tests/npu/a2a3/comm/st/testcase",  # 相对于 pto-isa root
    ],
}
```

## 测试结果

8 卡 TREDUCE (256 float 元素，reduce sum)：

```
=== ALL 8 RANKS PASSED ===
Sample: [2800.0, 2808.0, 2816.0, 2824.0, 2832.0, ...]
```
