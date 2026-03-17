/**
 * distributed_worker — per-device process for multi-card collective kernels.
 *
 * Handles HCCL initialization (RootInfo exchange, communicator, RDMA window
 * allocation) then delegates kernel execution to simpler's AICPU orchestration
 * via dlopen("libhost_runtime.so").
 *
 * PTO communication instructions access GVA addresses (windowsIn[]) via MTE2
 * DMA and do not require HCCL-bound stream; simpler uses its own stream.
 *
 * File-based IPC replaces MPI:
 *   - Rank 0 generates HcclRootInfo and writes to a shared file
 *   - Other ranks poll until the file appears
 *
 * Usage:
 *   distributed_worker --device-id 0 --rank 0 --nranks 8 \
 *       --artifact-dir build/artifacts \
 *       --rootinfo-file build/artifacts/rootinfo.bin \
 *       [--root 0] [--orch-func build_treduce_graph]
 *
 * Exit codes:
 *   0 — success
 *   1 — argument error
 *   2 — dlopen / dlsym / artifact error
 *   3 — ACL / HCCL error
 *   4 — simpler runtime error
 *   5 — verification error
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

#include "hccl_context.h"

// Internal HCCL APIs (not in public headers)
extern "C" HcclResult HcclAllocComResourceByTiling(
    HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(
    const char *group, HcclComm *commHandle);

using CommTopo = uint32_t;
extern "C" HcclResult HcomGetL0TopoTypeEx(
    const char *group, CommTopo *topoType, uint32_t isSetDevice);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

// Runtime stream APIs (minimal subset for HCCL init)
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" int32_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" int32_t rtStreamDestroy(rtStream_t stream);

// ============================================================================
// HCCL tiling structures (required by HcclAllocComResourceByTiling)
// ============================================================================

static constexpr uint32_t MAX_CC_TILING_NUM = 8U;
static constexpr uint32_t GROUP_NAME_SIZE = 128U;
static constexpr uint32_t ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

// RING topology compat structures for extracting windowsIn from HcclOpResParam
namespace hccl_compat {

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

// ============================================================================
// Constants and helpers
// ============================================================================

static constexpr size_t TREDUCE_COUNT = 256;
static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

static inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes) {
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

static inline void HcclHostBarrier(HcclComm comm, aclrtStream stream) {
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

// simpler C API function pointer types
using fn_get_runtime_size_t = size_t (*)();
using fn_set_device_t = int (*)(int);
using fn_init_runtime_t = int (*)(void*,
    const uint8_t*, size_t, const char*,
    uint64_t*, int, int*, uint64_t*,
    const int*, const uint8_t* const*, const size_t*, int);
using fn_launch_runtime_t = int (*)(void*,
    int, int, int,
    const uint8_t*, size_t,
    const uint8_t*, size_t,
    int);
using fn_finalize_runtime_t = int (*)(void*);

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static bool wait_for_file(const std::string& path, int timeout_sec = 120) {
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            auto sz = f.seekg(0, std::ios::end).tellg();
            if (sz >= static_cast<std::streamoff>(HCCL_ROOT_INFO_BYTES)) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static void file_barrier(const std::string& dir, int rank, int nranks,
                          const std::string& tag) {
    std::string my_marker = dir + "/barrier_" + tag + "_" +
                            std::to_string(rank) + ".ready";
    { std::ofstream(my_marker) << "1"; }

    for (int r = 0; r < nranks; ++r) {
        std::string marker = dir + "/barrier_" + tag + "_" +
                             std::to_string(r) + ".ready";
        while (true) {
            std::ifstream f(marker);
            if (f.good()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    int device_id = -1;
    int rank = -1;
    int nranks = -1;
    int root = 0;
    std::string artifact_dir;
    std::string rootinfo_file;
    std::string orch_func = "build_treduce_graph";

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--device-id" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg == "--rank" && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else if (arg == "--nranks" && i + 1 < argc) {
            nranks = std::atoi(argv[++i]);
        } else if (arg == "--root" && i + 1 < argc) {
            root = std::atoi(argv[++i]);
        } else if (arg == "--artifact-dir" && i + 1 < argc) {
            artifact_dir = argv[++i];
        } else if (arg == "--rootinfo-file" && i + 1 < argc) {
            rootinfo_file = argv[++i];
        } else if (arg == "--orch-func" && i + 1 < argc) {
            orch_func = argv[++i];
        }
    }

    if (device_id < 0 || rank < 0 || nranks <= 0 ||
        artifact_dir.empty() || rootinfo_file.empty()) {
        fprintf(stderr,
            "Usage: distributed_worker --device-id N --rank R --nranks N "
            "--artifact-dir PATH --rootinfo-file PATH "
            "[--root R] [--orch-func NAME]\n");
        return 1;
    }

    fprintf(stderr, "[rank %d] Starting: device=%d, nranks=%d, root=%d, "
            "orch_func=%s\n", rank, device_id, nranks, root, orch_func.c_str());

    // ========================================================================
    // Phase 0: Load simpler shared library
    // ========================================================================
    std::string lib_path = artifact_dir + "/libhost_runtime.so";
    void* lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "[rank %d] dlopen failed: %s\n", rank, dlerror());
        return 2;
    }

    auto fn_get_runtime_size = (fn_get_runtime_size_t)dlsym(lib, "get_runtime_size");
    auto fn_set_device       = (fn_set_device_t)dlsym(lib, "set_device");
    auto fn_init_runtime     = (fn_init_runtime_t)dlsym(lib, "init_runtime");
    auto fn_launch_runtime   = (fn_launch_runtime_t)dlsym(lib, "launch_runtime");
    auto fn_finalize_runtime = (fn_finalize_runtime_t)dlsym(lib, "finalize_runtime");

    if (!fn_get_runtime_size || !fn_set_device || !fn_init_runtime ||
        !fn_launch_runtime || !fn_finalize_runtime) {
        fprintf(stderr, "[rank %d] dlsym failed: %s\n", rank, dlerror());
        dlclose(lib);
        return 2;
    }

    // ========================================================================
    // Phase 1: ACL init
    // ========================================================================
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        fprintf(stderr, "[rank %d] aclInit failed: %d\n", rank, (int)aRet);
        dlclose(lib);
        return 3;
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[rank %d] aclrtSetDevice(%d) failed: %d\n",
                rank, device_id, (int)aRet);
        dlclose(lib);
        return 3;
    }
    fprintf(stderr, "[rank %d] ACL device %d set\n", rank, device_id);

    // ========================================================================
    // Phase 2: HCCL RootInfo exchange
    // ========================================================================
    HcclRootInfo rootInfo{};
    if (rank == 0) {
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            fprintf(stderr, "[rank 0] HcclGetRootInfo failed: %d\n", (int)hret);
            return 3;
        }
        std::ofstream fout(rootinfo_file, std::ios::binary);
        fout.write(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        fout.close();
        fprintf(stderr, "[rank 0] RootInfo written to %s (%u bytes)\n",
                rootinfo_file.c_str(), HCCL_ROOT_INFO_BYTES);
    } else {
        fprintf(stderr, "[rank %d] Waiting for rootinfo file ...\n", rank);
        if (!wait_for_file(rootinfo_file)) {
            fprintf(stderr, "[rank %d] Timeout waiting for rootinfo file\n", rank);
            return 3;
        }
        std::ifstream fin(rootinfo_file, std::ios::binary);
        fin.read(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        fprintf(stderr, "[rank %d] RootInfo loaded\n", rank);
    }

    // ========================================================================
    // Phase 3: HCCL communicator init + resource allocation
    // ========================================================================
    rtStream_t hccl_stream = nullptr;
    rtStreamCreate(&hccl_stream, RT_STREAM_PRIORITY_DEFAULT);

    fprintf(stderr, "[rank %d] HcclCommInitRootInfo (nranks=%d) ...\n", rank, nranks);
    HcclComm comm = nullptr;
    HcclResult hret = HcclCommInitRootInfo(
        static_cast<uint32_t>(nranks), &rootInfo,
        static_cast<uint32_t>(rank), &comm);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[rank %d] HcclCommInitRootInfo failed: %d\n",
                rank, (int)hret);
        return 3;
    }
    fprintf(stderr, "[rank %d] HCCL comm initialized\n", rank);

    char group[128] = {};
    hret = HcclGetCommName(comm, group);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[rank %d] HcclGetCommName failed: %d\n",
                rank, (int)hret);
        return 3;
    }

    CommTopo topoType = 0;
    hret = HcomGetL0TopoTypeEx(group, &topoType, COMM_IS_NOT_SET_DEVICE);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[rank %d] HcomGetL0TopoTypeEx failed: %d\n",
                rank, (int)hret);
        return 3;
    }
    fprintf(stderr, "[rank %d] Topology: %s\n", rank,
            topoType == COMM_TOPO_MESH ? "MESH" : "RING");

    HcclComm commHandle = nullptr;
    hret = HcomGetCommHandleByGroup(group, &commHandle);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[rank %d] HcomGetCommHandleByGroup failed: %d\n",
                rank, (int)hret);
        return 3;
    }

    file_barrier(artifact_dir, rank, nranks, "hccl_init");

    Mc2CommConfigV2 tiling{};
    memset(&tiling, 0, sizeof(tiling));
    tiling.init.version = 100U;
    tiling.init.mc2HcommCnt = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(&tiling.inner) -
        reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1);
    strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh",
            ALG_CONFIG_SIZE - 1);

    void *ctxPtr = nullptr;
    hret = HcclAllocComResourceByTiling(commHandle, hccl_stream,
                                         &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
        fprintf(stderr, "[rank %d] HcclAllocComResourceByTiling failed: %d\n",
                rank, (int)hret);
        return 3;
    }
    fprintf(stderr, "[rank %d] HCCL resources allocated, ctxPtr=%p\n",
            rank, ctxPtr);

    // Extract HcclDeviceContext (MESH vs RING)
    HcclDeviceContext hostCtx{};
    HcclDeviceContext *deviceCtx = nullptr;
    bool ownsDeviceCtx = false;

    if (topoType == COMM_TOPO_MESH) {
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx,
                           sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] MESH: aclrtMemcpy deviceCtx failed: %d\n",
                    rank, (int)aRet);
            return 3;
        }
    } else {
        using namespace hccl_compat;
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff,
                           sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] RING: read head failed: %d\n",
                    rank, (int)aRet);
            return 3;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);
        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes,
                           rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] RING: read remoteRes failed: %d\n",
                    rank, (int)aRet);
            return 3;
        }

        memset(&hostCtx, 0, sizeof(hostCtx));

        uint64_t wsFields[2] = {0, 0};
        aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        hostCtx.workSpace = wsFields[0];
        hostCtx.workSpaceSize = wsFields[1];
        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                continue;
            }
            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                fprintf(stderr, "[rank %d] RING: remoteRes[%u] is null\n",
                        rank, i);
                return 3;
            }
            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo),
                               reinterpret_cast<void*>(devPtr),
                               sizeof(remoteInfo), ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                fprintf(stderr, "[rank %d] RING: read remote info %u failed: %d\n",
                        rank, i, (int)aRet);
                return 3;
            }
            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
        }

        void *newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext),
                           ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] RING: malloc deviceCtx failed: %d\n",
                    rank, (int)aRet);
            return 3;
        }
        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx,
                           sizeof(HcclDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] RING: copy deviceCtx failed: %d\n",
                    rank, (int)aRet);
            aclrtFree(newDevMem);
            return 3;
        }
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(newDevMem);
        ownsDeviceCtx = true;
    }

    fprintf(stderr, "[rank %d] HCCL context: rankId=%u, rankNum=%u, "
            "winSize=%lu\n", rank, hostCtx.rankId, hostCtx.rankNum,
            hostCtx.winSize);
    for (uint32_t i = 0; i < hostCtx.rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
        fprintf(stderr, "[rank %d]   windowsIn[%u] = 0x%lx\n",
                rank, i, hostCtx.windowsIn[i]);
    }

    // ========================================================================
    // Phase 4: Place input data in HCCL window
    // ========================================================================
    uint64_t localWinBase = hostCtx.windowsIn[rank];
    size_t winOffset = 0;
    if (nranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }
    void *input_win_ptr = WindowAlloc(localWinBase, winOffset,
                                       TREDUCE_COUNT * sizeof(float));

    std::vector<float> input_host(TREDUCE_COUNT);
    for (size_t i = 0; i < TREDUCE_COUNT; ++i) {
        input_host[i] = static_cast<float>(i + rank * 100);
    }

    void *staging = nullptr;
    aRet = aclrtMalloc(&staging, TREDUCE_COUNT * sizeof(float),
                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[rank %d] aclrtMalloc staging failed: %d\n",
                rank, (int)aRet);
        return 3;
    }
    aclrtMemcpy(staging, TREDUCE_COUNT * sizeof(float),
                input_host.data(), TREDUCE_COUNT * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(input_win_ptr, TREDUCE_COUNT * sizeof(float),
                staging, TREDUCE_COUNT * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtFree(staging);

    void *output_dev = nullptr;
    aRet = aclrtMalloc(&output_dev, TREDUCE_COUNT * sizeof(float),
                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[rank %d] aclrtMalloc output failed: %d\n",
                rank, (int)aRet);
        return 3;
    }

    fprintf(stderr, "[rank %d] Data placed in HCCL window, input_ptr=%p, "
            "output=%p\n", rank, input_win_ptr, output_dev);

    HcclHostBarrier(comm, (aclrtStream)hccl_stream);

    // ========================================================================
    // Phase 5: Execute kernel via simpler AICPU orchestration
    // ========================================================================
    {
    int rc = fn_set_device(device_id);
    if (rc != 0) {
        fprintf(stderr, "[rank %d] simpler set_device failed: %d\n", rank, rc);
        dlclose(lib);
        return 4;
    }
    fprintf(stderr, "[rank %d] simpler DeviceRunner initialized\n", rank);

    auto orch_so  = read_file(artifact_dir + "/treduce_orch.so");
    auto aicpu_so = read_file(artifact_dir + "/libaicpu_kernel.so");
    auto aicore_o = read_file(artifact_dir + "/aicore_kernel.o");
    auto k_bin    = read_file(artifact_dir + "/treduce_kernel.bin");

    if (orch_so.empty() || aicpu_so.empty() || aicore_o.empty() ||
        k_bin.empty()) {
        fprintf(stderr, "[rank %d] Failed to load one or more artifacts from "
                "%s\n", rank, artifact_dir.c_str());
        dlclose(lib);
        return 2;
    }

    uint64_t func_args[5] = {
        reinterpret_cast<uint64_t>(input_win_ptr),
        reinterpret_cast<uint64_t>(output_dev),
        static_cast<uint64_t>(nranks),
        static_cast<uint64_t>(root),
        reinterpret_cast<uint64_t>(deviceCtx),
    };
    int arg_types[5] = {0, 0, 0, 0, 0};
    uint64_t arg_sizes[5] = {0, 0, 0, 0, 0};
    int kernel_func_ids[1] = {0};
    const uint8_t* kernel_ptrs[1] = {k_bin.data()};
    size_t kernel_sizes[1] = {k_bin.size()};

    size_t rt_size = fn_get_runtime_size();
    void* runtime = std::malloc(rt_size);
    if (!runtime) {
        fprintf(stderr, "[rank %d] malloc runtime failed\n", rank);
        dlclose(lib);
        return 4;
    }

    fprintf(stderr, "[rank %d] Initializing simpler runtime ...\n", rank);
    rc = fn_init_runtime(runtime,
        orch_so.data(), orch_so.size(), orch_func.c_str(),
        func_args, 5, arg_types, arg_sizes,
        kernel_func_ids, kernel_ptrs, kernel_sizes, 1);

    if (rc != 0) {
        fprintf(stderr, "[rank %d] simpler init_runtime failed: %d\n", rank, rc);
        std::free(runtime);
        dlclose(lib);
        return 4;
    }

    fprintf(stderr, "[rank %d] Launching kernel via simpler ...\n", rank);
    rc = fn_launch_runtime(runtime,
        1, 1, device_id,
        aicpu_so.data(), aicpu_so.size(),
        aicore_o.data(), aicore_o.size(), 0);

    if (rc != 0) {
        fprintf(stderr, "[rank %d] simpler launch_runtime failed: %d\n",
                rank, rc);
        std::free(runtime);
        dlclose(lib);
        return 4;
    }

    rc = fn_finalize_runtime(runtime);
    if (rc != 0) {
        fprintf(stderr, "[rank %d] simpler finalize_runtime failed: %d\n",
                rank, rc);
        std::free(runtime);
        dlclose(lib);
        return 4;
    }
    std::free(runtime);
    }

    fprintf(stderr, "[rank %d] Kernel execution complete\n", rank);
    HcclHostBarrier(comm, (aclrtStream)hccl_stream);

    // ========================================================================
    // Phase 6: Verify results (root only)
    // ========================================================================
    bool passed = true;
    if (rank == root) {
        std::vector<float> output_host(TREDUCE_COUNT);
        aRet = aclrtMemcpy(output_host.data(), TREDUCE_COUNT * sizeof(float),
                           output_dev, TREDUCE_COUNT * sizeof(float),
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            fprintf(stderr, "[rank %d] Copy output back failed: %d\n",
                    rank, (int)aRet);
            passed = false;
        } else {
            int mismatches = 0;
            for (size_t i = 0; i < TREDUCE_COUNT; ++i) {
                float expected = static_cast<float>(
                    nranks * static_cast<int>(i) +
                    100 * nranks * (nranks - 1) / 2);
                if (std::fabs(output_host[i] - expected) > 1e-3f) {
                    if (mismatches < 5) {
                        fprintf(stderr,
                            "[rank %d] MISMATCH [%zu]: got %.6f, expected %.6f\n",
                            rank, i, output_host[i], expected);
                    }
                    mismatches++;
                }
            }
            if (mismatches > 0) {
                fprintf(stderr, "[rank %d] FAILED — %d/%zu mismatches\n",
                        rank, mismatches, TREDUCE_COUNT);
                passed = false;
            } else {
                fprintf(stderr,
                    "\n=========================================="
                    "==============\n");
                fprintf(stderr,
                    "[rank %d] TREDUCE PASSED — all %zu elements correct!\n",
                    rank, TREDUCE_COUNT);
                fprintf(stderr, "Sample: [%.1f, %.1f, %.1f, %.1f, %.1f, ...]\n",
                        output_host[0], output_host[1], output_host[2],
                        output_host[3], output_host[4]);
                fprintf(stderr,
                    "=========================================="
                    "==============\n\n");
            }
        }
    }

    // ========================================================================
    // Phase 7: Cleanup
    // ========================================================================
    aclrtFree(output_dev);
    if (ownsDeviceCtx && deviceCtx) aclrtFree(deviceCtx);
    dlclose(lib);

    if (hccl_stream) rtStreamDestroy(hccl_stream);
    HcclCommDestroy(comm);
    aclrtResetDevice(device_id);
    aclFinalize();

    fprintf(stderr, "[rank %d] Done (status=%s)\n", rank,
            passed ? "PASS" : "FAIL");
    return passed ? 0 : 5;
}
