cmake_minimum_required(VERSION 3.16)
project(simpler_distributed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Platform selection: onboard (real CANN) or sim (host-only) ----
set(DISTRIBUTED_PLATFORM "onboard" CACHE STRING
    "Platform backend: onboard (ACL/HCCL) or sim (host shared-memory)")
set_property(CACHE DISTRIBUTED_PLATFORM PROPERTY STRINGS onboard sim)

message(STATUS "DISTRIBUTED_PLATFORM = ${DISTRIBUTED_PLATFORM}")

# ---- distributed_worker (common source) ----
add_executable(distributed_worker src/distributed_worker.cpp)

target_include_directories(distributed_worker PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_options(distributed_worker PRIVATE
    -Wall -Wextra -Wno-pedantic -Wno-stringop-truncation)

# ---- Platform-specific backend ----
if(DISTRIBUTED_PLATFORM STREQUAL "onboard")
    # Ascend CANN SDK required
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    if(NOT ASCEND_HOME_PATH)
        message(FATAL_ERROR
            "ASCEND_HOME_PATH not set. Source CANN set_env.sh first.")
    endif()

    find_path(ACL_INCLUDE_DIR acl/acl.h
        PATHS "${ASCEND_HOME_PATH}/include" NO_DEFAULT_PATH)
    find_library(ACL_LIBRARY ascendcl
        PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    find_library(HCCL_LIBRARY hccl
        PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    find_library(HCCL_FWK_LIBRARY hccl_fwk
        PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    find_library(RUNTIME_LIBRARY runtime
        PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)

    if(NOT ACL_INCLUDE_DIR OR NOT ACL_LIBRARY OR NOT HCCL_LIBRARY
       OR NOT HCCL_FWK_LIBRARY)
        message(FATAL_ERROR
            "ACL/HCCL libraries not found under ${ASCEND_HOME_PATH}")
    endif()

    message(STATUS "ACL: ${ACL_INCLUDE_DIR} / ${ACL_LIBRARY}")
    message(STATUS "HCCL: ${HCCL_LIBRARY}")

    target_sources(distributed_worker PRIVATE src/onboard_backend.cpp)
    target_include_directories(distributed_worker PRIVATE "${ACL_INCLUDE_DIR}")
    target_link_libraries(distributed_worker
        "${ACL_LIBRARY}" "${HCCL_LIBRARY}" "${HCCL_FWK_LIBRARY}" dl pthread)

    if(RUNTIME_LIBRARY)
        target_link_libraries(distributed_worker "${RUNTIME_LIBRARY}")
    endif()

elseif(DISTRIBUTED_PLATFORM STREQUAL "sim")
    # Pure host build — no CANN dependency
    target_sources(distributed_worker PRIVATE src/sim_backend.cpp)
    target_link_libraries(distributed_worker dl pthread rt)

else()
    message(FATAL_ERROR
        "Unknown DISTRIBUTED_PLATFORM: ${DISTRIBUTED_PLATFORM}. "
        "Use 'onboard' or 'sim'.")
endif()
