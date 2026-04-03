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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "dist_orchestrator.h"
#include "dist_ring.h"
#include "dist_scheduler.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

// ---------------------------------------------------------------------------
// MockWorker: run() blocks until complete() is called by the test thread.
// WorkerThread wraps it, so the Scheduler calls WorkerThread.dispatch() and
// WorkerThread calls MockWorker.run() in its own thread.
// ---------------------------------------------------------------------------

struct MockWorker : public IWorker {
    struct Record {
        DistTaskSlot slot;
        WorkerType type;
    };

    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    std::atomic<bool> is_running{false};

    void run(const WorkerPayload &p) override {
        {
            std::lock_guard<std::mutex> lk(dispatched_mu);
            dispatched.push_back({p.task_slot, p.worker_type});
        }
        is_running.store(true, std::memory_order_release);

        std::unique_lock<std::mutex> lk(run_mu);
        run_cv.wait(lk, [this] {
            return should_complete.load(std::memory_order_acquire);
        });
        should_complete.store(false, std::memory_order_relaxed);
        is_running.store(false, std::memory_order_release);
    }

    void complete() {
        std::lock_guard<std::mutex> lk(run_mu);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    // Wait until run() starts (dispatched and executing)
    void wait_running(int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!is_running.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int dispatched_count() {
        std::lock_guard<std::mutex> lk(dispatched_mu);
        return static_cast<int>(dispatched.size());
    }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct SchedulerFixture : public ::testing::Test {
    static constexpr int32_t N = DIST_TASK_WINDOW_SIZE;

    std::unique_ptr<DistTaskSlotState[]> slots;
    DistTensorMap tm;
    DistRing ring;
    DistScope scope;
    DistReadyQueue rq;
    DistOrchestrator orch;
    MockWorker chip_worker;
    DistScheduler sched;

    std::vector<DistTaskSlot> consumed_slots;
    std::mutex consumed_mu;

    void SetUp() override {
        slots = std::make_unique<DistTaskSlotState[]>(N);
        ring.init(N);
        orch.init(&tm, &ring, &scope, &rq, slots.get(), N);

        DistScheduler::Config cfg;
        cfg.slots = slots.get();
        cfg.num_slots = N;
        cfg.ready_queue = &rq;
        cfg.chip_workers = {&chip_worker};
        cfg.on_consumed_cb = [this](DistTaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(cfg);
    }

    void TearDown() override {
        sched.stop();
        ring.shutdown();
    }

    DistSubmitResult submit_chip(const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs) {
        WorkerPayload p;
        p.worker_type = WorkerType::CHIP;
        return orch.submit(WorkerType::CHIP, p, inputs, outputs);
    }

    void wait_consumed(DistTaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (DistTaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(SchedulerFixture, IndependentTaskDispatchedAndConsumed) {
    auto res = submit_chip({}, {{64}});
    DistTaskSlot slot = res.task_slot;

    // WorkerThread calls MockWorker.run() — wait for it to start
    chip_worker.wait_running();
    ASSERT_GE(chip_worker.dispatched_count(), 1);
    EXPECT_EQ(chip_worker.dispatched[0].slot, slot);

    // Signal completion → WorkerThread pushes to completion_queue → Scheduler consumes
    chip_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto a = submit_chip({}, {{128}});
    uint64_t a_key = reinterpret_cast<uint64_t>(a.outputs[0].ptr);

    auto b = submit_chip({{a_key}}, {{64}});
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::PENDING);

    // Complete A → B should become ready
    chip_worker.wait_running();
    EXPECT_EQ(chip_worker.dispatched[0].slot, a.task_slot);
    chip_worker.complete();  // A done

    // Wait for B to be dispatched
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (chip_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(chip_worker.dispatched_count(), 2);
    EXPECT_EQ(chip_worker.dispatched[1].slot, b.task_slot);

    chip_worker.complete();  // B done
    wait_consumed(b.task_slot);
}
