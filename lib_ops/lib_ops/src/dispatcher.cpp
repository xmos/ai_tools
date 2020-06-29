// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "lib_ops/api/device_memory.h"

namespace xcore {

static Dispatcher *kDispatcher = nullptr;

Dispatcher *GetDispatcher() {
  assert(kDispatcher);
  kDispatcher->ResetScratchAllocation();
  return kDispatcher;
}

XCoreStatus InitializeXCore(Dispatcher *dispatcher) {
  kDispatcher = dispatcher;
  return kXCoreOk;
}

#ifdef XCORE

#define IS_RAM(a) (((uintptr_t)a >= 0x80000) && ((uintptr_t)a <= 0x100000))

// xCORE Dispatcher implementation.
// Uses a threadgroup_t to dispatch tasks to threads.
Dispatcher::Dispatcher(void *buffer, size_t buffer_size,
                       bool use_current_thread)
    : use_current_thread_(use_current_thread) {
  group_ = thread_group_alloc();

  allocator_.SetHeap(buffer, buffer_size);

  tasks_.size = 0;
}

Dispatcher::~Dispatcher() { thread_group_free(group_); }

XCoreStatus Dispatcher::JoinTasks() {
  if (tasks_.size == 0) return kXCoreOk;

  int begin = 0;

  if (use_current_thread_) {
    (tasks_.function)(tasks_.arguments[begin]);
    begin++;
  }

  if (tasks_.stack == nullptr) {
    tasks_.stack_words += 2;
    tasks_.stack = reinterpret_cast<char *>(allocator_.AllocateScratchBuffer(
        tasks_.stack_words * bytes_per_stackword * (tasks_.size - begin),
        DOUBLE_WORD_ALIGNMENT));
  }

  for (int i = begin; i < tasks_.size; i++) {
    int32_t stack_offset =
        tasks_.stack_words * bytes_per_stackword * (i - begin);
    thread_group_add(
        group_, tasks_.function, tasks_.arguments[i],
        stack_base(&tasks_.stack[stack_offset], tasks_.stack_words));
  }

  thread_group_start(group_);
  thread_group_wait(group_);

  tasks_.size = 0;

  return kXCoreOk;
}

void Dispatcher::PreloadBuffer(int8_t **dest, int8_t const *src, int32_t size) {
  if (IS_RAM(src)) {
    *dest = (int8_t *)src;
  } else {
    if (*dest == nullptr) *dest = (int8_t *)AllocateScratchBuffer(size);

    memload((void **)dest, (void *)src, size);
  }
}

void Dispatcher::PreloadWeights(int8_t **dest, int8_t const *src, int32_t size,
                                ChannelGroup const &changrp) {
  if (IS_RAM(src)) {
    *dest = (int8_t *)&src[changrp.start * size];
  } else {
    if (*dest == nullptr)
      *dest = (int8_t *)AllocateScratchBuffer(changrp.size * size);

    memload((void **)dest, (void *)&src[changrp.start * size],
            changrp.size * size);
  }
}

void Dispatcher::PreloadBiases(int16_t **dest, int16_t const *src,
                               ChannelGroup const &changrp) {
  if (IS_RAM(src)) {
    *dest = (int16_t *)&src[changrp.index * bso_changrp_len];
  } else {
    if (*dest == nullptr)
      *dest = (int16_t *)AllocateScratchBuffer(bso_changrp_bytes);

    memload((void **)dest, (void *)&src[changrp.index * bso_changrp_len],
            bso_changrp_bytes);
  }
}

#else

// x86 Dispatcher implementation.
// Uses a std::vector of std::thread to dispatch tasks to threads.
Dispatcher::Dispatcher(void *buffer, size_t buffer_size,
                       bool use_current_thread)
    : use_current_thread_(use_current_thread) {
  allocator_.SetHeap(buffer, buffer_size);

  tasks_.size = 0;
}

Dispatcher::~Dispatcher() {}

XCoreStatus Dispatcher::JoinTasks() {
  if (tasks_.size == 0) return kXCoreOk;

  int begin = 0;

  if (use_current_thread_) {
    (tasks_.function)(tasks_.arguments[begin]);
    begin++;
  }

  // Start threads
  for (int i = begin; i < tasks_.size; i++) {
    group_.push_back(std::thread(tasks_.function, tasks_.arguments[i]));
  }

  // Join threads
  for (auto &thread : group_) {
    thread.join();
  }
  group_.clear();
  tasks_.size = 0;

  return kXCoreOk;
}

void Dispatcher::PreloadBuffer(int8_t **dest, int8_t const *src, int32_t size) {
  *dest = (int8_t *)src;

  // if (*dest == nullptr) *dest = (int8_t *)AllocateScratchBuffer(size);

  // std::memcpy((void **)*dest, (void *)src, size);
}

void Dispatcher::PreloadWeights(int8_t **dest, int8_t const *src, int32_t size,
                                ChannelGroup const &changrp) {
  *dest = (int8_t *)&src[changrp.start * size];

  // if (*dest == nullptr)
  //   *dest = (int8_t *)AllocateScratchBuffer(changrp.size * size);

  // std::memcpy((void **)*dest, (void *)&src[changrp.start * size],
  //             changrp.size * size);
}

void Dispatcher::PreloadBiases(int16_t **dest, int16_t const *src,
                               ChannelGroup const &changrp) {
  *dest = (int16_t *)&src[changrp.index * bso_changrp_len];

  // if (*dest == nullptr)
  //   *dest = (int16_t *)AllocateScratchBuffer(bso_changrp_bytes);

  // std::memcpy((void **)*dest, (void *)&src[changrp.index * bso_changrp_len],
  //             bso_changrp_bytes);
}

#endif  // XCORE

//**************************************
//**************************************
//**************************************
// Dispatcher methods common to
//   XCORE & x86
//**************************************
//**************************************
//**************************************

XCoreStatus Dispatcher::Reset() {
  tasks_.size = 0;
  allocator_.ResetHeap();

  return kXCoreOk;
}

XCoreStatus Dispatcher::InitializeTasks(thread_function_t function,
                                        size_t stack_words) {
  tasks_.function = function;
  tasks_.stack_words = stack_words;
  tasks_.size = 0;
  tasks_.stack = nullptr;

  return kXCoreOk;
}

void *Dispatcher::AllocatePersistantBuffer(size_t size, size_t alignment) {
  return allocator_.AllocatePersistantBuffer(size, alignment);
}

void *Dispatcher::AllocateScratchBuffer(size_t size, size_t alignment) {
  return allocator_.AllocateScratchBuffer(size, alignment);
}

XCoreStatus Dispatcher::ResetScratchAllocation() {
  allocator_.ResetScratch();

  return kXCoreOk;
}

size_t Dispatcher::GetMaxAllocatedSize() {
  return allocator_.GetMaxAllocatedSize();
}

XCoreStatus Dispatcher::AddTask(void *argument) {
  assert(tasks_.size < max_threads);

  if (tasks_.size < max_threads) {
    tasks_.arguments[tasks_.size] = argument;
    tasks_.size++;

    return kXCoreOk;
  }

  return kXCoreError;
}

}  // namespace xcore
