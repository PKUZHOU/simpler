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
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - TaskOutputTensors: Return value from submit containing materialized output Tensors
 * - Arg: Aggregated argument container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_

#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "pto_cq_types.h"     // NOLINT(build/include_subdir)
#include "pto_submit_types.h" // NOLINT(build/include_subdir) -- PTO2LaunchSpec
#include "task_args.h"        // NOLINT(build/include_subdir) -- TaskArgs base class
#include "tensor.h"           // NOLINT(build/include_subdir)
#include "tensor_arg.h"       // NOLINT(build/include_subdir) -- canonical TensorArgType definition

// Task arguments
#define MAX_TENSOR_ARGS 16   // Maximum tensor arguments per task
#define MAX_SCALAR_ARGS 128  // Maximum scalar arguments per task
#define PTO2_MAX_OUTPUTS 16  // Maximum outputs per task
#define PTO2_MAX_INPUTS 16   // Maximum inputs per task
#define PTO2_MAX_INOUTS 8    // Maximum in-out args per task
#define PTO2_MAX_COMPLETIONS_PER_TASK PTO2_CQ_MAX_ENTRIES

typedef enum {
    PTO2_ASYNC_ENGINE_SDMA = 0,
    PTO2_ASYNC_ENGINE_ROCE = 1,
    PTO2_ASYNC_ENGINE_URMA = 2,
    PTO2_ASYNC_ENGINE_CCU = 3,
    PTO2_NUM_ASYNC_ENGINES = 4
} PTO2AsyncEngine;

enum class PTO2CompletionType : int32_t {
    COUNTER = 0,
};

// =============================================================================
// Task Output Tensors (return value from submit)
// =============================================================================

/**
 * TaskOutputTensors — returned by submit, holds materialized output Tensors.
 *
 * Only runtime-created outputs are stored here, indexed in add_output order.
 *
 * The underlying storage is uninitialized; only output_count elements are
 * valid after submit returns.  This avoids default-constructing Tensor[]
 * on the hot path (2 KB of unnecessary zeroing per submit).
 *
 * Users must hold a named TaskOutputTensors variable and borrow via get_ref();
 * binding get_ref() on an rvalue is compile-time rejected to prevent dangling.
 */
class TaskOutputTensors {
public:  // NOLINT(whitespace/indent)
    TaskOutputTensors() :
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    const Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const Tensor &get_ref(uint32_t index) const && = delete;

    void materialize_output(const Tensor &tensor) {
        always_assert(output_count_ < PTO2_MAX_OUTPUTS);
        tensors_[output_count_++] = &tensor;
    }

private:  // NOLINT(whitespace/indent)
    uint32_t output_count_;
    const Tensor *tensors_[PTO2_MAX_OUTPUTS];
};

// =============================================================================
// Argument Types (for pto_submit_task API)
// =============================================================================

union TensorRef {
    const Tensor *ptr;
    const TensorCreateInfo *create_info;
    TensorRef() :
        ptr(nullptr) {}
};

template <typename T>
inline uint64_t pack_scalar_to_u64(T value) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "pack_scalar_to_u64: type must fit in 8 bytes");
    union {
        uint64_t u;
        T v;
    } packed{};
    packed.u = 0;
    packed.v = value;
    return packed.u;
}

/**
 * Aggregated argument container for pto_submit_task API.
 *
 * This keeps the PR444 `Arg` transport model, plus async completion metadata
 * used by deferred task submission helpers.
 */
struct Arg : TaskArgs<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType> {
    bool has_error{false};
    const char *error_msg{nullptr};
    PTO2LaunchSpec launch_spec;
    bool complete_in_future{false};
    uint64_t cq_addr{0};

    void reset() {
        clear();
        has_error = false;
        error_msg = nullptr;
        complete_in_future = false;
        cq_addr = 0;
    }

    void set_error(const char *msg) {
        if (!has_error) {
            has_error = true;
            error_msg = msg;
        }
    }

    bool check_add_tensor_valid() {
        if (scalar_count_ != 0) {
            set_error(
                "add_input/add_output/add_inout called after add_scalar: "
                "all tensors must be added before any scalars"
            );
            return false;
        }
        if (tensor_count_ >= MAX_TENSOR_ARGS) {
            set_error("Too many tensor args (exceeds MAX_TENSOR_ARGS=16)");
            return false;
        }
        return true;
    }

    void add_input(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::INPUT;
        tensor_count_++;
    }

    void add_output(const TensorCreateInfo &ci) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].create_info = &ci;
        tags_[tensor_count_] = TensorArgType::OUTPUT;
        tensor_count_++;
    }

    void add_output(TensorCreateInfo &&) = delete;

    void add_inout(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::INOUT;
        tensor_count_++;
    }

    void add_output(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::OUTPUT_EXISTING;
        tensor_count_++;
    }

    void add_no_dep(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::NO_DEP;
        tensor_count_++;
    }

    template <typename T = uint64_t>
    void add_scalar(T value) {
        if (scalar_count_ >= MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        scalars_[scalar_count_++] = pack_scalar_to_u64(value);
    }

    void add_scalars(const uint64_t *values, int count) {
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        memcpy(&scalars_[scalar_count_], values, count * sizeof(uint64_t));
        scalar_count_ += count;
    }

    void add_scalars_i32(const int32_t *values, int count) {
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        uint64_t *dst = &scalars_[scalar_count_];
#if defined(__aarch64__)
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t *>(values + i));
            uint64x2_t lo = vmovl_u32(vget_low_u32(v));
            uint64x2_t hi = vmovl_u32(vget_high_u32(v));
            vst1q_u64(dst + i, lo);
            vst1q_u64(dst + i + 2, hi);
        }
        for (; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#else
        for (int i = 0; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#endif
        scalar_count_ += count;
    }

    void copy_scalars_from(const Arg &src, int src_offset, int count) {
        if (src_offset + count > src.scalar_count_) {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        memcpy(&scalars_[scalar_count_], &src.scalars_[src_offset], count * sizeof(uint64_t));
        scalar_count_ += count;
    }
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
