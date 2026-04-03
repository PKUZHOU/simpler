# Communication Mechanism Porting to PR444

## Background

This document records how the communication-related implementation was ported onto the current `tensormap_and_ringbuffer` runtime baseline and brought up on real distributed hardware.

The work was based on two upstream inputs:

- `simpler` PR #444: the implementation baseline that introduced the current PTO2-style runtime path, including device-side orchestration, `Arg`-based task submission, runtime ops indirection, and deferred completion support.
- `pypto_top_level_design_documents` PR #2: the design baseline that clarified the role split between Host, AICPU, AICore, task graph submission, and runtime layering.

The goal of the port was not to reintroduce the old communication stack unchanged. The goal was to express the existing communication mechanism in the execution model defined by PR #444 and make distributed async communication work end-to-end.

## What Was Being Ported

The port covered two communication capabilities:

1. Distributed runtime bring-up across ranks, including remote window discovery and per-rank worker launch.
2. Asynchronous communication completion, including:
   - async remote read via `TGET_ASYNC`
   - notification-based synchronization via `TNOTIFY`
   - deferred task completion on AICPU after the kernel has already returned

In the current implementation, Host is responsible for preparing the distributed execution environment. The actual communication operation is executed on device by AICore kernels using remote addresses from `CommDeviceContext`.

## Baseline from PR444

PR #444 already provided the runtime structure that the communication path had to plug into:

- Orchestration submits tasks through `pto_orchestration_api.h`.
- Runtime owns a task graph, dependency tracking, ready queues, and scheduler state.
- AICPU runs the orchestrator and scheduler.
- AICore executes dispatched kernels.
- Deferred completion is represented by `Arg::complete_in_future` and `Arg::cq_addr`.

Key files from the baseline:

- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

This meant the port did not need to invent a new async model. It needed to attach communication semantics to the deferred-completion hooks that PR #444 already exposed.

## Porting Strategy

The port was done in four layers.

### 1. Rebuild the distributed host-side launch path around PR444

PR #444 changed the runtime calling convention and artifact packaging, so the old distributed launch path could not be reused directly.

The new flow is:

1. `examples/scripts/distributed_code_runner.py` compiles:
   - host runtime shared library
   - AICPU binary
   - AICore binary
   - orchestration `.so`
   - child kernel `.bin`
2. It writes all artifacts into `build/distributed/artifacts`.
3. It spawns one `distributed_worker.py` process per rank.
4. Each worker initializes communication, allocates distributed windows, loads inputs, wraps binaries into `ChipCallable`, and launches `ChipWorker.run(...)`.

Key files:

- `examples/scripts/distributed_code_runner.py`
- `examples/scripts/distributed_worker.py`

### 2. Reconnect distributed memory addressing to the PR444 runtime

The old communication logic relied on every rank being able to reconstruct the peer-visible address of a symmetric window allocation.

That contract is now represented explicitly by `CommDeviceContext`:

```c++
struct CommDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];
};
```

Key file:

- `src/a2a3/platform/include/common/comm_context.h`

`src/a2a3/platform/onboard/host/comm_hccl.cpp` was then used as the Host-side bridge:

- initialize HCCL communicator
- allocate communication resources
- extract or reconstruct remote window addresses
- materialize `CommDeviceContext`
- copy the context to device memory
- pass the device pointer into orchestration arguments

This is the point where distributed Host initialization becomes visible to device-side kernels.

### 3. Map communication operations onto PR444 task submission

Once `CommDeviceContext*` became available as an orchestration argument, the communication kernels could be expressed as regular PTO2 tasks.

The important adaptation was:

- communication kernels are submitted like normal AIV tasks
- their arguments include `CommDeviceContext*`
- remote addresses are derived on device from `windowsIn[]` plus local offset

The helper pattern used in the kernels is:

```c++
template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx,
                                      __gm__ T* local_ptr,
                                      int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}
```

This preserved the previous communication semantics while fitting the PR444 task model.

Examples:

- `examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/kernels/aiv/kernel_producer_async.cpp`
- `examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/kernels/aiv/kernel_producer_notify.cpp`

### 4. Re-express async communication as deferred completion

This was the most important conceptual part of the port.

Before the port, the communication path already needed a notion of "kernel returns first, communication finishes later". PR #444 provided exactly that through deferred completion.

The mapping was:

| Communication requirement | PR444 runtime mechanism |
| --- | --- |
| kernel launches async transfer or waits on notify | submit task with `complete_in_future = true` |
| kernel needs to tell runtime what it is waiting on | write completion conditions into CQ |
| runtime must not release downstream tasks too early | AICPU scheduler polls CQ-derived conditions before completing task |

## How Deferred Completion Was Connected

### CQ-based async completion for `TGET_ASYNC`

The port added a CQ-backed path for kernels that launch async DMA or remote reads.

Host/runtime side:

- orchestration allocates a CQ through `pto2_rt_alloc_cq()`
- it submits the producer through `pto2_rt_submit_aiv_task_deferred(...)`
- the runtime records `complete_in_future` and appends `cq_addr` as the last scalar kernel argument

Key files:

- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`

Device side:

- the producer kernel reads `cq_addr`
- launches `TGET_ASYNC`
- converts the returned event into a CQ entry
- flushes the CQ before returning

Key file:

- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_cq_kernel_api.h`

Concrete example:

- `examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/kernels/orchestration/async_demo_orchestration.cpp`
- `examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/kernels/aiv/kernel_producer_async.cpp`

### Counter-based deferred completion for `TNOTIFY`

The notification case uses the same deferred-completion framework, but the completion condition is a local counter threshold rather than an async DMA event handle.

The mapping is:

- producer sends `TNOTIFY(AtomicAdd)` to peer counter
- a dedicated `notify_wait` task is submitted as deferred
- that task produces a token tensor
- downstream consumer takes the token tensor as an input dependency
- AICPU only marks `notify_wait` complete when `*counter >= expected_value`

Key file:

- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_notify_kernel_api.h`

Concrete example:

- `examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/kernels/orchestration/async_notify_orchestration.cpp`

This let the port reuse one runtime completion model for both:

- event-backed async transfer completion
- counter-backed notification synchronization

## Scheduler-Side Bring-Up

After the kernel returns, the task is not always complete.

The scheduler path works like this:

1. AICPU observes that the dispatched subtask has returned.
2. It calls `on_subtask_complete(...)`.
3. If the whole mixed task is done, it checks whether the task is deferred.
4. If the task is deferred, it reads the kernel-written CQ and registers wait conditions in `PTO2AsyncWaitList`.
5. Each scheduler loop iteration polls those conditions first.
6. Only after all conditions are satisfied does AICPU call `on_mixed_task_complete(...)` and release downstream tasks.

Key files:

- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_async_wait.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

This is the core reason the async communication path now fits PR #444 cleanly: communication completion is no longer hard-coded into Host or kernel return semantics. It is represented as scheduler-visible runtime state.

## Main Files Touched During the Port

### Distributed launch and Host communication context

- `examples/scripts/distributed_code_runner.py`
- `examples/scripts/distributed_worker.py`
- `src/a2a3/platform/onboard/host/comm_hccl.cpp`

### Runtime async plumbing

- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_async_wait.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

### Example validation cases

- `examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/...`
- `examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/...`

## The Real Bring-Up Bug

The final blocker was not in `TGET_ASYNC`, `TNOTIFY`, or CQ polling logic.

The real bug was in runtime kernel dispatch:

- `runtime_maker.cpp` uploaded each child kernel as a `CoreCallable`
- the runtime stored the base address of the `CoreCallable` envelope
- RT2 dispatch payload expected the executable entry address instead
- AICore jumped into the header rather than the kernel code
- the observed result on hardware was an illegal-instruction style failure

The fix was to store:

```c++
callable_addr + CoreCallable::binary_data_offset()
```

instead of `callable_addr`.

Key file:

- `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`

This bug is worth calling out because it looks like an async communication failure from the outside, but it is actually a dispatch ABI mismatch between the Host runtime and RT2 AICore execution path.

## Validation Path

Two examples were used as acceptance tests for the port.

### 1. Async completion demo

Purpose:

- verify cross-rank remote read
- verify deferred completion via CQ
- verify AICPU scheduler only releases consumer after async event completion

Command shape:

```bash
source <your-ascend-env-script>
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/golden.py \
  -p a2a3 \
  --devices <dev0>,<dev1> \
  --build
```

### 2. Async notify demo

Purpose:

- verify inter-rank notification through `TNOTIFY`
- verify counter-based deferred completion
- verify token-based dependency gating of downstream consumer

Command shape:

```bash
source <your-ascend-env-script>
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/golden.py \
  -p a2a3 \
  --devices <dev0>,<dev1> \
  --build
```

These two examples cover the two async communication styles used in the port.

## What "Ported Successfully" Means in This Runtime

The communication mechanism should be considered successfully ported onto PR #444 only when all of the following are true:

1. Distributed worker launch can initialize per-rank communication context and pass `CommDeviceContext*` to orchestration.
2. Device kernels can derive peer addresses from the distributed window map.
3. Async communication kernels can return before communication completion without breaking task dependency semantics.
4. AICPU scheduler can observe deferred completion conditions and release downstream tasks only after they are truly satisfied.
5. Validation examples pass on real two-card hardware.

## Practical Lessons

### 1. Keep Host and device responsibilities separate

Host should prepare communication context and runtime payloads. Device should own actual communication issue and completion semantics.

### 2. Treat async completion as runtime state, not as kernel control flow

Once async completion is represented as scheduler-visible state, the same runtime mechanism can support DMA events and notification counters.

### 3. Verify dispatch ABI early

If distributed async communication appears broken, confirm the dispatched AICore entry address before investigating communication semantics.

### 4. Use example kernels as executable specs

`async_completion_demo` and `async_notify_demo` are not only tests. They are also the clearest reference implementations for future communication ports.

## Recommended Checklist for Future Communication Ports

When porting another communication primitive onto this runtime, use this order:

1. Confirm the Host can materialize the required distributed context and pass it into orchestration.
2. Express the primitive as a normal PTO2 task first, without async behavior.
3. If the primitive completes later than kernel return, convert it to deferred completion.
4. Decide whether completion should be represented by:
   - event/CQ entry
   - counter threshold
5. Add a focused distributed example before integrating into a larger operator.
6. Validate on real hardware before debugging higher-level orchestration logic.

