// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "lib_ops/api/allocator.h"
#include "lib_ops/api/planning.h"

namespace xcore {

constexpr size_t bytes_per_stackword = 4;

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
  int begin = 0;
  char *stack = nullptr;

  if (use_current_thread_) {
    const Task &task = tasks_.data[begin];
    (task.function)(task.argument);
    begin++;
  }

  stack = reinterpret_cast<char *>(allocator_.AllocateScratchBuffer(
      tasks_.stack_words * bytes_per_stackword * (tasks_.size - begin)));

  for (int i = begin; i < tasks_.size; i++) {
    const Task &task = tasks_.data[i];
    int32_t stack_offset = tasks_.stack_words * bytes_per_stackword * i;
    thread_group_add(group_, task.function, task.argument,
                     stack_base(&stack[stack_offset], tasks_.stack_words));
  }

  thread_group_start(group_);
  thread_group_wait(group_);

  tasks_.size = 0;

  return kXCoreOk;
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

XCoreStatus Dispatcher::InitializeTasks(thread_function_t function,
                                        size_t stack_words) {
  tasks_.function = function;
  tasks_.stack_words = stack_words;
  tasks_.size = 0;

  return kXCoreOk;
}

XCoreStatus Dispatcher::Reset() {
  tasks_.size = 0;
  allocator_.ResetHeap();

  return kXCoreOk;
}

void *Dispatcher::AllocatePersistantBuffer(size_t size) {
  return allocator_.AllocatePersistantBuffer(size);
}

void *Dispatcher::AllocateScratchBuffer(size_t size) {
  return allocator_.AllocateScratchBuffer(size);
}

XCoreStatus Dispatcher::ResetScratchAllocation() {
  allocator_.ResetScratch();

  return kXCoreOk;
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
