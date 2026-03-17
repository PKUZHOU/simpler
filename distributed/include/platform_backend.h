#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "hccl_context.h"

struct PlatformBackend {
    virtual ~PlatformBackend() = default;

    // Device management
    virtual int init() = 0;
    virtual int set_device(int device_id) = 0;
    virtual int reset_device(int device_id) = 0;
    virtual int finalize() = 0;

    // Memory management — "device" memory on onboard, host memory on sim
    virtual int malloc_device(void** ptr, size_t size) = 0;
    virtual void free_device(void* ptr) = 0;
    virtual int memcpy_h2d(void* dst, const void* src, size_t size) = 0;
    virtual int memcpy_d2h(void* dst, const void* src, size_t size) = 0;
    virtual int memcpy_d2d(void* dst, const void* src, size_t size) = 0;

    // Communication — HCCL on onboard, shared-memory mock on sim
    virtual int comm_init(int nranks, int rank,
                          const std::string& rootinfo_file,
                          const std::string& artifact_dir) = 0;
    virtual int comm_alloc_resources() = 0;
    virtual void comm_barrier() = 0;
    virtual int comm_destroy() = 0;

    // Window / context access (valid after comm_alloc_resources)
    virtual HcclDeviceContext* get_device_context() = 0;
    virtual const HcclDeviceContext& get_host_context() const = 0;
};

std::unique_ptr<PlatformBackend> create_backend(const std::string& platform);
