// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "lib_ops/api/allocator.h"
#include "lib_ops/api/par.h"

namespace xcore {

constexpr size_t bytes_per_stackword = 4;

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
// xCORE Dispatcher implementation.
// Uses a threadgroup_t to dispatch tasks to threads.
Dispatcher::Dispatcher(void *buffer, size_t size, int num_threads,
                       bool use_current_thread)
    : num_threads_(num_threads), use_current_thread_(use_current_thread) {
  group_ = thread_group_alloc();

  xcSetHeap(buffer, size);

  // Allocate TaskArray
  tasks_.data = reinterpret_cast<Task *>(xcMalloc(sizeof(Task) * num_threads_));
  tasks_.size = 0;
}

Dispatcher::~Dispatcher() { thread_group_free(group_); }

XCoreStatus Dispatcher::Join() {
  int begin = 0;
  char *stack = nullptr;

  if (use_current_thread_) {
    const Task &task = tasks_.data[begin];
    (task.function)(task.argument);
    begin++;
  }

  stack = reinterpret_cast<char *>(xcMalloc(
      tasks_.stack_words * bytes_per_stackword * (tasks_.size - begin)));

  for (int i = begin; i < tasks_.size; i++) {
    const Task &task = tasks_.data[i];
    int32_t stack_offset = tasks_.stack_words * bytes_per_stackword * i;
    thread_group_add(group_, task.function, task.argument,
                     stack_base(&stack[stack_offset], tasks_.stack_words));
  }

  thread_group_start(group_);
  thread_group_wait(group_);

  xcFree(stack);

  return kXCoreOk;
}

#else
// x86 Dispatcher implementation.
// Uses a std::vector of std::thread to dispatch tasks to threads.
Dispatcher::Dispatcher(void *buffer, size_t size, int num_threads,
                       bool use_current_thread)
    : num_threads_(num_threads), use_current_thread_(use_current_thread) {
  xcSetHeap(buffer, size);

  // Allocate TaskArray
  tasks_.data = reinterpret_cast<Task *>(xcMalloc(sizeof(Task) * num_threads_));
  tasks_.size = 0;
}

Dispatcher::~Dispatcher() {}

XCoreStatus Dispatcher::Join() {
  int begin = 0;
  if (use_current_thread_) {
    const Task &task = tasks_.data[begin];
    (task.function)(task.argument);
    begin++;
  }

  // Start threads
  for (int i = begin; i < tasks_.size; i++) {
    const Task &task = tasks_.data[i];
    group_.push_back(std::thread(task.function, task.argument));
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

XCoreStatus Dispatcher::Reset() {
  tasks_.size = 0;
  xcResetHeap();

  return kXCoreOk;
}

XCoreStatus Dispatcher::AddThread(thread_function_t function, void *argument,
                                  size_t stack_words) {
  assert(tasks_.size < num_threads_);

  if (tasks_.size < num_threads_) {
    tasks_.stack_words = std::max(tasks_.stack_words, stack_words);
    tasks_.data[tasks_.size] = {function, (void *)argument};
    tasks_.size++;

    return kXCoreOk;
  }

  return kXCoreError;
}

}  // namespace xcore
