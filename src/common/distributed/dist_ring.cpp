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

#include "dist_ring.h"

#include <stdexcept>

void DistRing::init(int32_t window_size) {
    if (window_size <= 0 || (window_size & (window_size - 1)) != 0)
        throw std::invalid_argument("DistRing window_size must be a positive power of 2");
    window_size_ = window_size;
    window_mask_ = window_size - 1;
    next_task_id_ = 0;
    last_alive_.store(-1, std::memory_order_relaxed);
    shutdown_ = false;
}

DistTaskSlot DistRing::alloc() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
        if (shutdown_) return true;
        // Active tasks = next_task_id_ - (last_alive_ + 1)
        // Allow alloc when active tasks < window_size_
        return (next_task_id_ - last_alive_.load(std::memory_order_acquire) - 1) < window_size_;
    });
    if (shutdown_) return DIST_INVALID_SLOT;
    int32_t task_id = next_task_id_++;
    return task_id & window_mask_;
}

void DistRing::release(DistTaskSlot slot) {
    // Derive which task_id this slot corresponds to.
    // last_alive tracks the highest released task_id (monotonically advancing).
    // We advance last_alive to at least the task_id that owns this slot.
    // Since slots are released roughly in order, this is safe.
    int32_t current = last_alive_.load(std::memory_order_acquire);
    // The slot belongs to some task_id; find the smallest task_id >= current+1
    // that maps to this slot.
    int32_t base = current + 1;
    int32_t offset = ((slot - base) & window_mask_);
    int32_t task_id = base + offset;

    int32_t expected = current;
    while (task_id > expected) {
        if (last_alive_.compare_exchange_weak(
                expected, task_id, std::memory_order_release, std::memory_order_relaxed
            )) {
            break;
        }
        // expected updated by CAS; retry if another thread advanced it past us
        if (expected >= task_id) break;
    }
    cv_.notify_all();
}

int32_t DistRing::active_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return next_task_id_ - last_alive_.load(std::memory_order_acquire) - 1;
}

void DistRing::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}
