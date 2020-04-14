// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "lib_ops/api/dispatcher.h"
#include "lib_ops/api/par.h"

namespace xcore {

constexpr size_t bytes_per_stackword = 4;

static Dispatcher* gDispatcher = nullptr;

Dispatcher* GetDispatcher() {
  assert(gDispatcher);
  return gDispatcher;
}

XCoreStatus InitializeXCore(Dispatcher* dispatcher) {
  gDispatcher = dispatcher;
  return kXCoreOk;
}

#ifdef XCORE
// xCORE Dispatcher implementation.
// Uses a threadgroup_t to dispatch kernel funnctions to HW threads.
Dispatcher::Dispatcher(void* buffer, size_t size, int num_threads,
                       bool use_current_thread)
    : num_threads_(num_threads),
      use_current_thread_(use_current_thread),
      stack_size_(0),
      stack_ptr_(nullptr) {
  group_ = thread_group_alloc();
  allocator_.SetBuffer(buffer, size);

  // Allocate TaskArray
  tasks_.data =
      reinterpret_cast<Task*>(allocator_.Allocate(sizeof(Task) * num_threads_));
  tasks_.size = 0;
}

Dispatcher::~Dispatcher() { thread_group_free(group_); }

void Dispatcher::Start() {
  int begin = 0;
  if (use_current_thread_) {
    const Task& task = tasks_.data[begin];
    (task.function)(task.argument);
    begin++;
  }

  for (int i = begin; i < tasks_.size; i++) {
    const Task& task = tasks_.data[i];
    thread_group_add(group_, task.function, task.argument,
                     stack_base(task.stack, task.stack_words));
  }

  thread_group_start(group_);
}

void Dispatcher::Wait() { thread_group_wait(group_); }

#else
// x86 Dispatcher implementation.
// Uses a std::vector of std::thread to dispatch kernel functions to SW
// threads.
Dispatcher::Dispatcher(void* buffer, size_t size, int num_threads,
                       bool use_current_thread)
    : num_threads_(num_threads),
      use_current_thread_(use_current_thread),
      stack_ptr_(nullptr) {
  allocator_.SetBuffer(buffer, size);

  // Allocate TaskArray
  tasks_.data =
      reinterpret_cast<Task*>(allocator_.Allocate(sizeof(Task) * num_threads_));
  tasks_.size = 0;
}

Dispatcher::~Dispatcher() {}

void Dispatcher::Start() {
  int begin = 0;
  if (use_current_thread_) {
    const Task& task = tasks_.data[begin];
    (task.function)(task.argument);
    begin++;
  }

  for (int i = begin; i < tasks_.size; i++) {
    const Task& task = tasks_.data[i];
    group_.push_back(std::thread(task.function, task.argument));
  }
}

void Dispatcher::Wait() {
  for (auto& thread : group_) {
    thread.join();
  }
  group_.clear();

  tasks_.size = 0;
}

#endif

XCoreStatus Dispatcher::Reset() {
  stack_size_ = 0;
  stack_ptr_ = nullptr;
  allocator_.Reset();

  return kXCoreOk;
}

void* Dispatcher::AllocateStackBuffer(int32_t num_threads, size_t stack_words) {
  assert(num_threads <= num_threads_);
  size_t stack_size = stack_words * bytes_per_stackword * num_threads;
  stack_size_ = std::max(stack_size, stack_size_);
  stack_ptr_ = allocator_.Reallocate(stack_ptr_, stack_size_);

  return stack_ptr_;
}

void* Dispatcher::AllocatePersistentBuffer(size_t size) {
  return allocator_.Allocate(size);
}

XCoreStatus Dispatcher::Add(thread_function_t function, void* argument,
                            size_t stack_words) {
  assert(stack_ptr_);
  assert(tasks_.size < num_threads_);

  int32_t offset = stack_words * bytes_per_stackword * tasks_.size;

  if (offset > stack_size_) {
    return kXCoreError;
  }

  void* stack = reinterpret_cast<char*>(stack_ptr_) + offset;
  tasks_.data[tasks_.size] = {function, (void*)argument, stack_words, stack};
  tasks_.size++;

  return kXCoreOk;
}

}  // namespace xcore
