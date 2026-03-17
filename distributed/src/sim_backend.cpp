#include "platform_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr size_t SIM_WINDOW_SIZE = 8 * 1024 * 1024; // 8 MB per rank

static std::string shm_name_for_rank(const std::string& session, int rank) {
    return "/simpler_dist_" + session + "_r" + std::to_string(rank);
}

static std::string session_from_path(const std::string& rootinfo_file) {
    size_t h = std::hash<std::string>{}(rootinfo_file);
    char buf[17];
    snprintf(buf, sizeof(buf), "%016zx", h);
    return std::string(buf);
}

// ============================================================================
// SimBackend: pure-host simulation with POSIX shared memory for RDMA windows
// ============================================================================

class SimBackend final : public PlatformBackend {
public:
    ~SimBackend() override { cleanup_shm(); }

    int init() override { return 0; }
    int set_device(int /*device_id*/) override { return 0; }
    int reset_device(int /*device_id*/) override { return 0; }
    int finalize() override { return 0; }

    int malloc_device(void** ptr, size_t size) override {
        *ptr = std::malloc(size);
        return *ptr ? 0 : -1;
    }

    void free_device(void* ptr) override { std::free(ptr); }

    int memcpy_h2d(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return 0;
    }

    int memcpy_d2h(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return 0;
    }

    int memcpy_d2d(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return 0;
    }

    int comm_init(int nranks, int rank,
                  const std::string& rootinfo_file,
                  const std::string& artifact_dir) override {
        rank_ = rank;
        nranks_ = nranks;
        artifact_dir_ = artifact_dir;
        session_ = session_from_path(rootinfo_file);

        // Rank 0 writes a dummy rootinfo file for synchronization
        if (rank == 0) {
            std::vector<char> dummy(4108, 0);
            std::ofstream fout(rootinfo_file, std::ios::binary);
            fout.write(dummy.data(), static_cast<std::streamsize>(dummy.size()));
        } else {
            for (int i = 0; i < 1200; ++i) {
                std::ifstream f(rootinfo_file, std::ios::binary);
                if (f.good() && f.seekg(0, std::ios::end).tellg() > 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (i == 1199) return -1;
            }
        }
        return 0;
    }

    int comm_alloc_resources() override {
        file_barrier("hccl_init");

        shm_ptrs_.resize(nranks_, nullptr);

        // Each rank creates its own segment, then opens all others
        std::string my_name = shm_name_for_rank(session_, rank_);
        int fd = shm_open(my_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) { perror("shm_open create"); return -1; }
        if (ftruncate(fd, static_cast<off_t>(SIM_WINDOW_SIZE)) < 0) {
            perror("ftruncate"); close(fd); return -1;
        }
        void *p = mmap(nullptr, SIM_WINDOW_SIZE, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) { perror("mmap"); return -1; }
        std::memset(p, 0, SIM_WINDOW_SIZE);
        shm_ptrs_[rank_] = p;

        file_barrier("shm_create");

        // Open other ranks' segments
        for (int r = 0; r < nranks_; ++r) {
            if (r == rank_) continue;
            std::string name = shm_name_for_rank(session_, r);
            int rfd = shm_open(name.c_str(), O_RDWR, 0666);
            if (rfd < 0) { perror("shm_open peer"); return -1; }
            void *rp = mmap(nullptr, SIM_WINDOW_SIZE, PROT_READ | PROT_WRITE,
                            MAP_SHARED, rfd, 0);
            close(rfd);
            if (rp == MAP_FAILED) { perror("mmap peer"); return -1; }
            shm_ptrs_[r] = rp;
        }

        // Build the simulated HcclDeviceContext
        memset(&host_ctx_, 0, sizeof(host_ctx_));
        host_ctx_.rankId = static_cast<uint32_t>(rank_);
        host_ctx_.rankNum = static_cast<uint32_t>(nranks_);
        host_ctx_.winSize = SIM_WINDOW_SIZE;
        for (int r = 0; r < nranks_; ++r) {
            host_ctx_.windowsIn[r] = reinterpret_cast<uint64_t>(shm_ptrs_[r]);
        }

        // device_ctx_ points to host_ctx_ (in sim, device == host)
        device_ctx_mem_ = std::malloc(sizeof(HcclDeviceContext));
        std::memcpy(device_ctx_mem_, &host_ctx_, sizeof(HcclDeviceContext));
        device_ctx_ = reinterpret_cast<HcclDeviceContext*>(device_ctx_mem_);

        file_barrier("shm_ready");
        return 0;
    }

    void comm_barrier() override {
        static int barrier_seq_ = 0;
        file_barrier("run_" + std::to_string(barrier_seq_++));
    }

    int comm_destroy() override {
        cleanup_shm();
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

    void cleanup_shm() {
        for (int r = 0; r < nranks_; ++r) {
            if (shm_ptrs_.size() > static_cast<size_t>(r) && shm_ptrs_[r]) {
                munmap(shm_ptrs_[r], SIM_WINDOW_SIZE);
                shm_ptrs_[r] = nullptr;
            }
        }
        // Only rank 0 unlinks shared memory
        if (rank_ == 0) {
            for (int r = 0; r < nranks_; ++r) {
                shm_unlink(shm_name_for_rank(session_, r).c_str());
            }
        }
        if (device_ctx_mem_) { std::free(device_ctx_mem_); device_ctx_mem_ = nullptr; }
        device_ctx_ = nullptr;
    }

    int rank_ = 0;
    int nranks_ = 1;
    std::string artifact_dir_;
    std::string session_;
    std::vector<void*> shm_ptrs_;
    HcclDeviceContext host_ctx_{};
    HcclDeviceContext* device_ctx_ = nullptr;
    void* device_ctx_mem_ = nullptr;
};

std::unique_ptr<PlatformBackend> create_backend(const std::string& /*platform*/) {
    return std::make_unique<SimBackend>();
}
