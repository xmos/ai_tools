// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "lib_ops/api/device_memory.h"
#include "lib_ops/api/planning.h"

namespace xcore {

static Dispatcher *kDispatcher = nullptr;

Dispatcher *GetDispatcher() {
  assert(kDispatcher);
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

  int remaining_tasks = tasks_.size - begin;

  if (remaining_tasks > 0) {
    if (tasks_.stack == nullptr) {
      tasks_.stack_words += 2;

      tasks_.stack = reinterpret_cast<char *>(allocator_.AllocateScratchBuffer(
          tasks_.stack_words * bytes_per_stackword * remaining_tasks,
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
  }

  tasks_.size = 0;
  allocator_.ResetScratch();

  return kXCoreOk;
}

#else

#define IS_RAM(a) (1)

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

size_t Dispatcher::GetAllocatedSize() { return allocator_.GetAllocatedSize(); }

uintptr_t Dispatcher::GetScratchBuffer() {
  return (uintptr_t)allocator_.GetScratchBuffer();
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

void Dispatcher::FetchBuffer(int8_t **dest, int8_t const *src, size_t size) {
  if (IS_RAM(src)) {
    *dest = (int8_t *)src;
  } else {
    memload((void **)dest, (void *)src, size);
  }
}

void Dispatcher::FetchWeights(int8_t **dest, int8_t const *src, size_t size,
                              ChannelGroup const &changrp) {
  size_t changrp_bytes = size / changrp_len;
  // changrp_bytes = 32;
  if (IS_RAM(src)) {
    *dest = (int8_t *)&src[changrp.start * changrp_bytes];
  } else {
    memload((void **)dest, (void *)&src[changrp.start * changrp_bytes],
            changrp.size * changrp_bytes);
  }
}

void Dispatcher::FetchBiases(int16_t **dest, int16_t const *src, size_t size,
                             ChannelGroup const &changrp) {
  if (IS_RAM(src)) {
    *dest = (int16_t *)&src[changrp.index * bso_changrp_len];
  } else {
    memload((void **)dest, (void *)&src[changrp.index * size], size);
  }
}

}  // namespace xcore
