# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HostWorker — L3 host-side orchestration worker.

HostWorker wraps DistWorker(level=3) and manages:
  - SubWorker processes (fork/shm, for Python callables)
  - ChipWorker threads (one per device, for NPU execution — wired in post-merge)
  - Automatic dependency tracking via TensorMap
  - Scope-based intermediate tensor lifetime management

Usage::

    hw = HostWorker(num_sub_workers=2)

    @hw.register
    def my_postprocess():
        ...

    hw.init()

    def my_orch(hw, _args):
        payload = WorkerPayload()
        payload.worker_type = WorkerType.SUB
        payload.callable_id = my_postprocess.callable_id
        hw.submit(WorkerType.SUB, payload)

    hw.execute(HostTask(orch=my_orch))
    hw.close()
"""

import ctypes
import os
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

from task_interface import (
    DIST_SUB_MAILBOX_SIZE,
    DistInputSpec,
    DistOutputSpec,
    DistSubmitResult,
    DistSubWorker,
    DistWorker,
    WorkerPayload,
    WorkerType,
)

from .host_task import HostTask

# Mailbox layout (must match dist_sub_worker.cpp offsets)
_OFF_STATE = 0  # int32: IDLE=0, TASK_READY=1, TASK_DONE=2, SHUTDOWN=3
_OFF_CALLABLE_ID = 4  # int32
_OFF_ERROR_CODE = 24  # int32

_IDLE = 0
_TASK_READY = 1
_TASK_DONE = 2
_SHUTDOWN = 3


def _mailbox_ptr(shm: SharedMemory) -> int:
    """Return the raw memory address of a SharedMemory buffer."""
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _sub_worker_loop(buf: memoryview, registry: dict) -> None:
    """Main loop for a forked SubWorker child process.

    Polls mailbox state and executes registered callables.
    Exits cleanly on SHUTDOWN.  Must be called in a child process created by
    os.fork() — uses os._exit() to avoid running atexit handlers.
    """
    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]

        if state == _TASK_READY:
            cid = struct.unpack_from("i", buf, _OFF_CALLABLE_ID)[0]
            fn = registry.get(cid)
            error = 0
            if fn is None:
                error = 1
            else:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    error = 2
            struct.pack_into("i", buf, _OFF_ERROR_CODE, error)
            # Release store: error_code written before state=TASK_DONE
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)

        elif state == _SHUTDOWN:
            break
        # Tight spin: same as L2 AICPU pattern (dedicated execution unit)


class HostWorker:
    """L3 host worker — thin Python wrapper over DistWorker(level=3).

    Lifecycle::

        hw = HostWorker(num_sub_workers=N)
        cid = hw.register(my_fn)   # register callables BEFORE init()
        hw.init()                  # forks SubWorkers, starts Scheduler
        hw.execute(task)           # run orch, drain
        hw.close()                 # stop Scheduler, reap SubWorkers

    Alternatively use as a context manager::

        with HostWorker(num_sub_workers=N) as hw:
            cid = hw.register(my_fn)
            hw.execute(task)
    """

    def __init__(self, num_sub_workers: int = 0) -> None:
        self._num_sub_workers = num_sub_workers
        self._callable_registry: dict[int, Callable] = {}
        self._shms: list[SharedMemory] = []
        self._pids: list[int] = []
        self._dist_worker: Optional[DistWorker] = None
        self._dist_sub_workers: list[DistSubWorker] = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Callable registration (must be called BEFORE init())
    # ------------------------------------------------------------------

    def register(self, fn: Callable) -> int:
        """Register a Python callable for use as a SUB task.

        Must be called before init() so the callable is inherited by forked
        child processes without pickling.  Returns the callable_id to pass
        in WorkerPayload.callable_id.
        """
        if self._initialized:
            raise RuntimeError("register() must be called before init()")
        cid = len(self._callable_registry)
        self._callable_registry[cid] = fn
        return cid

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Fork SubWorker processes and start the C++ Scheduler thread.

        fork() is called BEFORE creating C++ threads (DistWorker.init()) to
        comply with POSIX fork-in-multithreaded-process restrictions.
        """
        if self._initialized:
            raise RuntimeError("HostWorker already initialized")

        # 1. Allocate shared-memory mailboxes (one per SubWorker)
        for _ in range(self._num_sub_workers):
            shm = SharedMemory(create=True, size=DIST_SUB_MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
            self._shms.append(shm)

        # 2. Fork SubWorker processes — must happen before any C++ thread starts
        registry = self._callable_registry  # COW snapshot for children
        for i in range(self._num_sub_workers):
            pid = os.fork()
            if pid == 0:
                # Child: run worker loop then exit cleanly
                buf = self._shms[i].buf
                assert buf is not None
                _sub_worker_loop(buf, registry)
                os._exit(0)  # skip atexit / pytest handlers
            else:
                self._pids.append(pid)

        # 3. Create DistWorker and wire sub-workers
        dw = DistWorker(3)
        self._dist_worker = dw

        for shm in self._shms:
            addr = _mailbox_ptr(shm)
            sub_w = DistSubWorker(addr)
            self._dist_sub_workers.append(sub_w)
            dw.add_sub_worker(sub_w)

        # 4. Start Scheduler (C++ threads start here, safely after fork)
        dw.init()
        self._initialized = True

    def close(self) -> None:
        """Stop the Scheduler and reap SubWorker processes."""
        if not self._initialized:
            return

        if self._dist_worker:
            self._dist_worker.close()
            self._dist_worker = None

        # Signal SubWorker processes to exit
        for shm in self._shms:
            buf = shm.buf
            assert buf is not None
            struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
        for pid in self._pids:
            os.waitpid(pid, 0)

        # Release shared memory
        for shm in self._shms:
            shm.close()
            shm.unlink()

        self._shms.clear()
        self._pids.clear()
        self._dist_sub_workers.clear()
        self._initialized = False

    # ------------------------------------------------------------------
    # Orchestration API (called from inside HostTask.orch)
    # ------------------------------------------------------------------

    def submit(
        self,
        worker_type: WorkerType,
        payload: WorkerPayload,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ) -> DistSubmitResult:
        """Submit a task to the distributed engine.

        Args:
            worker_type: WorkerType.CHIP or WorkerType.SUB.
            payload:     WorkerPayload with callable/args filled in.
            inputs:      List of tensor base_ptr (uint64) for dependency lookup.
            outputs:     List of output byte sizes for allocation.

        Returns:
            DistSubmitResult with task_slot and allocated output buffer pointers.
        """
        assert self._dist_worker is not None
        in_specs = [DistInputSpec(p) for p in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        return self._dist_worker.submit(worker_type, payload, in_specs, out_specs)

    def scope_begin(self) -> None:
        assert self._dist_worker is not None
        self._dist_worker.scope_begin()

    def scope_end(self) -> None:
        assert self._dist_worker is not None
        self._dist_worker.scope_end()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, task: HostTask) -> None:
        """Run the orchestration function, then wait for all tasks to complete.

        No drain() is exposed — waiting is internal to execute(), mirroring L2.
        """
        assert self._initialized and self._dist_worker is not None
        task.orch(self, task.args)
        self._dist_worker.drain()  # GIL released in C++

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HostWorker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
