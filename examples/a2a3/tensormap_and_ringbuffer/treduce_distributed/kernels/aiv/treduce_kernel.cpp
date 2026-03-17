/**
 * TREDUCE kernel for simpler's kernel_entry signature.
 *
 * Performs collective reduce (Sum) across multiple NPU ranks using PTO comm
 * instructions. Each rank's input data resides in an HCCL RDMA window;
 * the root rank gathers and sums all inputs into the output buffer.
 *
 * PTO communication instructions access remote data through GVA addresses
 * (windowsIn[]) via MTE2 DMA over HCCS; no HCCL-bound stream is required.
 *
 * args layout (all uint64_t, cast as needed):
 *   args[0] = __gm__ float* input   (device addr in HCCL window)
 *   args[1] = __gm__ float* output  (device addr, regular allocation)
 *   args[2] = int nranks
 *   args[3] = int root
 *   args[4] = __gm__ HcclDeviceContext* ctx  (device addr)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "hccl_context.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t TREDUCE_COUNT = 256;

template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(
    __gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* input  = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[1]);
    int nranks = static_cast<int>(args[2]);
    int root   = static_cast<int>(args[3]);
    __gm__ HcclDeviceContext* hcclCtx =
        reinterpret_cast<__gm__ HcclDeviceContext*>(args[4]);

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC,
                                  pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC,
                                   pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn,
                                         pto::Layout::ND>;
    using TileData  = pto::Tile<pto::TileType::Vec, float, 1, TREDUCE_COUNT,
                                 pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, TREDUCE_COUNT);
    StrideDyn stride(TREDUCE_COUNT, TREDUCE_COUNT, TREDUCE_COUNT,
                     TREDUCE_COUNT, 1);

    TileData accTile(1, TREDUCE_COUNT);
    TileData recvTile(1, TREDUCE_COUNT);
    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    if (my_rank == root) {
        Global outputG(output, shape, stride);
        Global tensors[16];
        int actual_nranks = (nranks > 16) ? 16 : nranks;
        for (int i = 0; i < actual_nranks; ++i) {
            __gm__ float* remoteInput = HcclRemotePtr(hcclCtx, input, i);
            tensors[i] = Global(remoteInput, shape, stride);
        }
        pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile,
                           pto::comm::ReduceOp::Sum);
    }

    pipe_barrier(PIPE_ALL);
}
