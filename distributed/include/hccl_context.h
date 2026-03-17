#pragma once

#include <cstdint>

// HcclDeviceContext — device-side communication context.
//
// On MESH topology (A2), HCCL returns a struct whose first fields match this
// layout directly — windowsIn[64] contains per-rank RDMA/GVA addresses.
// On RING topology (A3), the host builds this struct by extracting remote
// RDMA addresses from HcclOpResParam's remoteRes array.
//
// PTO communication instructions (TREDUCE, TGET, TPUT) access remote data
// through these GVA addresses via MTE2 DMA; no HCCL-bound stream is required.

static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;

struct HcclDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
};
