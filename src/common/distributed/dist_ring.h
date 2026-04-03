/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * DistRing — task slot allocator with back-pressure.
 *
 * Maintains a circular window of DIST_TASK_WINDOW_SIZE slots.  The Orchestrator
 * calls alloc() to claim the next slot before submitting a task.  The Scheduler
 * calls release() when a task reaches CONSUMED, advancing last_alive so the
 * Orchestrator can progress.
 *
 * Back-pressure: alloc() blocks (condition_variable wait) when the window is
 * full, i.e. when (next_task_id_ - last_alive_) >= window_size_.  This mirrors
 * L2's spin-wait but uses std::condition_variable to avoid burning host CPU.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include "dist_types.h"

class DistRing {
public:
    void init(int32_t window_size = DIST_TASK_WINDOW_SIZE);

    // Allocate next slot.  Blocks until space is available.
    // Returns the slot index (task_id % window_size).
    DistTaskSlot alloc();

    // Release slot.  Called by Scheduler when task reaches CONSUMED.
    // Advances last_alive so alloc() can proceed.
    void release(DistTaskSlot slot);

    int32_t window_size() const { return window_size_; }
    int32_t active_count() const;

private:
    int32_t window_size_{DIST_TASK_WINDOW_SIZE};
    int32_t window_mask_{DIST_TASK_WINDOW_SIZE - 1};
    int32_t next_task_id_{0};              // orch-only, no atomic needed
    std::atomic<int32_t> last_alive_{-1};  // updated by Scheduler

    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};

public:
    void shutdown();
};
