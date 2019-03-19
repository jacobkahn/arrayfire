/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <af/memory.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CPU_MEM_DEBUG
#define AF_CPU_MEM_DEBUG 0
#endif

#ifndef AF_CUDA_MEM_DEBUG
#define AF_CUDA_MEM_DEBUG 0
#endif

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

namespace spdlog {
class logger;
}
namespace common {
using mutex_t      = std::mutex;
using lock_guard_t = std::lock_guard<mutex_t>;

const unsigned MAX_BUFFERS = 1000;
const size_t ONE_GB        = 1 << 30;

// Global functions for setting a C af_memory_manager's internal ptrs
int MemoryManagerCWrapper_getActiveDeviceId(af_memory_manager* impl_);

int MemoryManagerCWrapper_getMaxMemorySize(af_memory_manager* impl_, int id);

void* MemoryManagerCWrapper_nativeAlloc(af_memory_manager* impl_, size_t size);

void MemoryManagerCWrapper_nativeFree(af_memory_manager* impl_, void* ptr);

// Wrap a C af_memory_manager struct
class MemoryManagerCWrapper : public af::MemoryManagerBase
{
    // A pointer to some memory manager
    af_memory_manager* impl_;
    
  public:
    ~MemoryManagerCWrapper() {
        // Like the C++ API, with the C API, the expectation is that
        // we control the managers themselves once they're passed, so free them.
        free(impl_);     
    }
  
    explicit MemoryManagerCWrapper(af_memory_manager* manager) {
      impl_ = manager;
    }

    void initialize() override {
      impl_->wrapper_handle = (af_memory_manager*)this;

      impl_->af_memory_manager_get_active_device_id =
        MemoryManagerCWrapper_getActiveDeviceId;
      impl_->af_memory_manager_get_max_memory_size =
        MemoryManagerCWrapper_getMaxMemorySize;
      impl_->af_memory_manager_native_alloc =
        MemoryManagerCWrapper_nativeAlloc;
      impl_->af_memory_manager_native_free =
        MemoryManagerCWrapper_nativeFree;
      
      impl_->af_memory_manager_initialize(impl_);
    }

    void shutdown() override {
      impl_->af_memory_manager_shutdown(impl_);
    }
    
    void addMemoryManagement(int device) override {
      impl_->af_memory_manager_add_memory_management(impl_, device);
    }

    void removeMemoryManagement(int device) override {
      impl_->af_memory_manager_remove_memory_management(impl_, device);
    }

    void* alloc(const size_t size, bool user_lock) override {
      return impl_->af_memory_manager_alloc(impl_, size, user_lock);
    }

    size_t allocated(void* ptr) override {
      return impl_->af_memory_manager_allocated(impl_, ptr);
    }

    void unlock(void* ptr, bool user_unlock) override {
      impl_->af_memory_manager_unlock(impl_, ptr, user_unlock);
    }

    void bufferInfo(size_t* alloc_bytes, size_t* alloc_buffers,
                    size_t* lock_bytes, size_t* lock_buffers) override {
      impl_->af_memory_manager_buffer_info(
        impl_,
        alloc_bytes,
        alloc_buffers,
        lock_bytes,
        lock_buffers
      );
    }

    void userLock(const void* ptr) override {
      impl_->af_memory_manager_user_lock(impl_, ptr);
    }

    void userUnlock(const void* ptr) override {
      impl_->af_memory_manager_user_unlock(impl_, ptr);
    }

    bool isUserLocked(const void* ptr) override {
      return impl_->af_memory_manager_is_user_locked(impl_, ptr);
    }

    bool checkMemoryLimit() override {
      return impl_->af_memory_manager_check_memory_limit(impl_);
    }

    size_t getMaxBytes() override {
      return impl_->af_memory_manager_get_max_bytes(impl_);
    }

    unsigned getMaxBuffers() override {
      return impl_->af_memory_manager_get_max_buffers(impl_);
    }

    void printInfo(const char* msg, const int device) override {
      impl_->af_memory_manager_print_info(impl_, msg, device);
    }

    void garbageCollect() override {
      impl_->af_memory_manager_garbage_collect(impl_);
    }

    size_t getMemStepSize() override {
      return impl_->af_memory_manager_get_mem_step_size(impl_);
    }

    void setMemStepSize(size_t new_step_size) override {
      impl_->af_memory_manager_set_mem_step_size(impl_, new_step_size);
    }
};

/**
 * The default ArrayFire memory manager.
 */
class MemoryManager : public af::MemoryManagerBase
{
    typedef struct {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    } locked_info;

    using locked_t    = typename std::unordered_map<void *, locked_info>;
    using locked_iter = typename locked_t::iterator;

    using free_t    = std::unordered_map<size_t, std::vector<void *>>;
    using free_iter = free_t::iterator;

    using uptr_t = std::unique_ptr<void, std::function<void(void *)>>;

    typedef struct memory_info {
        locked_t locked_map;
        free_t free_map;

        size_t lock_bytes;
        size_t lock_buffers;
        size_t total_bytes;
        size_t total_buffers;
        size_t max_bytes;

        memory_info() {
            // Calling getMaxMemorySize() here calls the virtual function that
            // returns 0 Call it from outside the constructor.
            max_bytes     = ONE_GB;
            total_bytes   = 0;
            total_buffers = 0;
            lock_bytes    = 0;
            lock_buffers  = 0;
        }
    } memory_info;

    size_t mem_step_size;
    unsigned max_buffers;
    
    std::shared_ptr<spdlog::logger> logger;
    bool debug_mode;
    mutex_t memory_mutex;

    memory_info& getCurrentMemoryInfo();

   public:
    ~MemoryManager() = default;
    MemoryManager() = default;
    MemoryManager(int num_devices, unsigned max_buffers, bool debug);

    // Initializes the memory manager
    virtual void initialize() override;

    // Shuts down the memory manager
    virtual void shutdown() override;

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void addMemoryManagement(int device) override;

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void removeMemoryManagement(int device) override;

    void setMaxMemorySize();

    /// Returns a pointer of size at least long
    ///
    /// This funciton will return a memory location of at least \p size
    /// bytes. If there is already a free buffer available, it will use
    /// that buffer. Otherwise, it will allocate a new buffer using the
    /// nativeAlloc function.
    void *alloc(const size_t size, bool user_lock) override;

    /// returns the size of the buffer at the pointer allocated by the memory
    /// manager.
    size_t allocated(void *ptr) override;

    /// Frees or marks the pointer for deletion during the nex garbage
    /// collection event
    void unlock(void *ptr, bool user_unlock) override;

    /// Frees all buffers which are not locked by the user or not being used.
    void garbageCollect() override;

    void printInfo(const char *msg, const int device) override;
    void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                    size_t *lock_bytes, size_t *lock_buffers) override;
    void userLock(const void *ptr) override;
    void userUnlock(const void *ptr) override;
    bool isUserLocked(const void *ptr) override;
    size_t getMemStepSize() override;
    size_t getMaxBytes() override;
    unsigned getMaxBuffers() override;
    void setMemStepSize(size_t new_step_size) override;
    bool checkMemoryLimit() override;

   protected:
    MemoryManager(const MemoryManager& other) = delete;
    MemoryManager(const MemoryManager&& other) = delete;
    MemoryManager& operator=(const MemoryManager& other) = delete;
    MemoryManager& operator=(const MemoryManager&& other) = delete;

    std::vector<memory_info> memory;

    void cleanDeviceMemoryManager(int device);
};

}  // namespace common
