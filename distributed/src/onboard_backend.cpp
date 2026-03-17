#include "platform_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

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

namespace hccl_compat {

struct HcclSignalInfo { uint64_t resId; uint64_t addr; uint32_t devId; uint32_t tsId; uint32_t rankId; uint32_t flag; };
struct HcclStreamInfo { int32_t streamIds; uint32_t sqIds; uint32_t cqIds; uint32_t logicCqids; };
struct ListCommon { uint64_t nextHost; uint64_t preHost; uint64_t nextDevice; uint64_t preDevice; };

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum; uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank; uint32_t userRankSize; int32_t deviceLogicId;
    bool isSingleMeshAggregation; uint32_t deviceNumPerAggregation;
    uint32_t superPodNum; uint32_t devicePhyId; uint32_t topoType;
    uint32_t deviceType; uint32_t serverNum; uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode; uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank; bool isDiffDeviceModule; bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation; uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum; uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum; uint64_t pairLinkCounter;
    uint32_t nicNum; uint64_t nicList; uint64_t complanRankLength; uint64_t complanRank;
    uint64_t bridgeRankNum; uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength; uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic; uint8_t retryEnable; uint8_t highPerfEnable;
    uint8_t padding[5]; uint8_t linkTimeOut[8]; uint64_t notifyWaitTime;
    uint32_t retryHoldTime; uint32_t retryIntervalTime; bool interXLinkDisable;
    uint32_t floatOverflowMode; uint32_t multiQpThreshold;
};

struct RemoteResPtr { uint64_t nextHostPtr; uint64_t nextDevicePtr; };
struct HcclMC2WorkSpace { uint64_t workspace; uint64_t workspaceSize; };

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId; uint32_t remoteWorldRank;
    uint64_t windowsIn; uint64_t windowsOut; uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId; uint32_t rankSize; uint64_t winSize;
    uint64_t localWindowsIn; uint64_t localWindowsOut;
    char hcomId[128]; uint64_t winExpSize; uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId; uint32_t rankSize; uint64_t winSize;
    uint64_t localWindowsIn; uint64_t localWindowsOut;
    char hcomId[128]; uint64_t winExpSize; uint64_t localWindowsExp;
    uint32_t rWinStart; uint32_t rWinOffset; uint64_t version;
    LocalResInfoV2 localRes; AlgoTopoInfo topoInfo; HcclOpConfig config;
    uint64_t hostStateInfo; uint64_t aicpuStateInfo; uint64_t lockAddr;
    uint32_t rsv[16]; uint32_t notifysize; uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

// ============================================================================
// OnboardBackend implementation
// ============================================================================

class OnboardBackend final : public PlatformBackend {
public:
    int init() override {
        constexpr int kAclRepeatInit = 100002;
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && static_cast<int>(ret) != kAclRepeatInit) {
            fprintf(stderr, "[onboard] aclInit failed: %d\n", (int)ret);
            return -1;
        }
        return 0;
    }

    int set_device(int device_id) override {
        device_id_ = device_id;
        aclError ret = aclrtSetDevice(device_id);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[onboard] aclrtSetDevice(%d) failed: %d\n",
                    device_id, (int)ret);
            return -1;
        }
        return 0;
    }

    int reset_device(int /*device_id*/) override {
        aclrtResetDevice(device_id_);
        return 0;
    }

    int finalize() override {
        aclFinalize();
        return 0;
    }

    int malloc_device(void** ptr, size_t size) override {
        aclError ret = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        return (ret == ACL_SUCCESS) ? 0 : -1;
    }

    void free_device(void* ptr) override {
        if (ptr) aclrtFree(ptr);
    }

    int memcpy_h2d(void* dst, const void* src, size_t size) override {
        return (aclrtMemcpy(dst, size, src, size,
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) ? 0 : -1;
    }

    int memcpy_d2h(void* dst, const void* src, size_t size) override {
        return (aclrtMemcpy(dst, size, src, size,
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) ? 0 : -1;
    }

    int memcpy_d2d(void* dst, const void* src, size_t size) override {
        return (aclrtMemcpy(dst, size, src, size,
                            ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS) ? 0 : -1;
    }

    int comm_init(int nranks, int rank,
                  const std::string& rootinfo_file,
                  const std::string& artifact_dir) override {
        rank_ = rank;
        nranks_ = nranks;
        artifact_dir_ = artifact_dir;

        rtStreamCreate(&hccl_stream_, RT_STREAM_PRIORITY_DEFAULT);

        HcclRootInfo rootInfo{};
        if (rank == 0) {
            HcclResult hret = HcclGetRootInfo(&rootInfo);
            if (hret != HCCL_SUCCESS) return -1;
            std::ofstream fout(rootinfo_file, std::ios::binary);
            fout.write(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
            fout.close();
            fprintf(stderr, "[rank 0] RootInfo written (%u bytes)\n",
                    HCCL_ROOT_INFO_BYTES);
        } else {
            for (int i = 0; i < 1200; ++i) {
                std::ifstream f(rootinfo_file, std::ios::binary);
                if (f.good()) {
                    auto sz = f.seekg(0, std::ios::end).tellg();
                    if (sz >= static_cast<std::streamoff>(HCCL_ROOT_INFO_BYTES)) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (i == 1199) return -1;
            }
            std::ifstream fin(rootinfo_file, std::ios::binary);
            fin.read(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        }

        HcclResult hret = HcclCommInitRootInfo(
            static_cast<uint32_t>(nranks), &rootInfo,
            static_cast<uint32_t>(rank), &comm_);
        if (hret != HCCL_SUCCESS) {
            fprintf(stderr, "[rank %d] HcclCommInitRootInfo failed: %d\n",
                    rank, (int)hret);
            return -1;
        }
        return 0;
    }

    int comm_alloc_resources() override {
        char group[128] = {};
        HcclResult hret = HcclGetCommName(comm_, group);
        if (hret != HCCL_SUCCESS) return -1;

        CommTopo topoType = 0;
        hret = HcomGetL0TopoTypeEx(group, &topoType, COMM_IS_NOT_SET_DEVICE);
        if (hret != HCCL_SUCCESS) return -1;

        HcclComm commHandle = nullptr;
        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) return -1;

        file_barrier("hccl_init");

        Mc2CommConfigV2 tiling{};
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
        hret = HcclAllocComResourceByTiling(commHandle, hccl_stream_,
                                             &tiling, &ctxPtr);
        if (hret != HCCL_SUCCESS || !ctxPtr) return -1;

        return extract_context(ctxPtr, topoType);
    }

    void comm_barrier() override {
        HcclBarrier(comm_, (aclrtStream)hccl_stream_);
        aclrtSynchronizeStream((aclrtStream)hccl_stream_);
    }

    int comm_destroy() override {
        if (hccl_stream_) rtStreamDestroy(hccl_stream_);
        if (comm_) HcclCommDestroy(comm_);
        if (owns_device_ctx_ && device_ctx_) free_device(device_ctx_);
        return 0;
    }

    HcclDeviceContext* get_device_context() override { return device_ctx_; }
    const HcclDeviceContext& get_host_context() const override { return host_ctx_; }

private:
    void file_barrier(const std::string& tag) {
        std::string my_marker = artifact_dir_ + "/barrier_" + tag + "_" +
                                std::to_string(rank_) + ".ready";
        { std::ofstream(my_marker) << "1"; }
        for (int r = 0; r < nranks_; ++r) {
            std::string marker = artifact_dir_ + "/barrier_" + tag + "_" +
                                 std::to_string(r) + ".ready";
            while (true) {
                std::ifstream f(marker);
                if (f.good()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    int extract_context(void* ctxPtr, CommTopo topoType) {
        memset(&host_ctx_, 0, sizeof(host_ctx_));

        if (topoType == COMM_TOPO_MESH) {
            device_ctx_ = reinterpret_cast<HcclDeviceContext*>(ctxPtr);
            aclError ret = aclrtMemcpy(&host_ctx_, sizeof(host_ctx_),
                                        device_ctx_, sizeof(host_ctx_),
                                        ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) return -1;
        } else {
            using namespace hccl_compat;
            auto *raw = reinterpret_cast<uint8_t*>(ctxPtr);

            HcclOpResParamHead head{};
            aclrtMemcpy(&head, sizeof(head),
                        raw + offsetof(HcclOpResParam, localUsrRankId),
                        sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);

            std::vector<RemoteResPtr> remotes(head.rankSize);
            size_t rb = head.rankSize * sizeof(RemoteResPtr);
            aclrtMemcpy(remotes.data(), rb,
                        raw + offsetof(HcclOpResParam, remoteRes), rb,
                        ACL_MEMCPY_DEVICE_TO_HOST);

            uint64_t ws[2] = {};
            aclrtMemcpy(ws, sizeof(ws), raw, sizeof(ws),
                        ACL_MEMCPY_DEVICE_TO_HOST);

            host_ctx_.workSpace = ws[0];
            host_ctx_.workSpaceSize = ws[1];
            host_ctx_.rankId = head.localUsrRankId;
            host_ctx_.rankNum = head.rankSize;
            host_ctx_.winSize = head.winSize;

            for (uint32_t i = 0; i < head.rankSize; ++i) {
                if (i == head.localUsrRankId) {
                    host_ctx_.windowsIn[i] = head.localWindowsIn;
                    continue;
                }
                uint64_t devPtr = remotes[i].nextDevicePtr;
                if (!devPtr) return -1;
                HcclRankRelationResV2 ri{};
                aclrtMemcpy(&ri, sizeof(ri), reinterpret_cast<void*>(devPtr),
                            sizeof(ri), ACL_MEMCPY_DEVICE_TO_HOST);
                host_ctx_.windowsIn[i] = ri.windowsIn;
            }

            void *newDev = nullptr;
            if (aclrtMalloc(&newDev, sizeof(HcclDeviceContext),
                            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return -1;
            if (aclrtMemcpy(newDev, sizeof(HcclDeviceContext), &host_ctx_,
                            sizeof(HcclDeviceContext),
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                aclrtFree(newDev);
                return -1;
            }
            device_ctx_ = reinterpret_cast<HcclDeviceContext*>(newDev);
            owns_device_ctx_ = true;
        }
        return 0;
    }

    int device_id_ = 0;
    int rank_ = 0;
    int nranks_ = 1;
    std::string artifact_dir_;
    rtStream_t hccl_stream_ = nullptr;
    HcclComm comm_ = nullptr;
    HcclDeviceContext host_ctx_{};
    HcclDeviceContext* device_ctx_ = nullptr;
    bool owns_device_ctx_ = false;
};

std::unique_ptr<PlatformBackend> create_backend(const std::string& /*platform*/) {
    return std::make_unique<OnboardBackend>();
}
