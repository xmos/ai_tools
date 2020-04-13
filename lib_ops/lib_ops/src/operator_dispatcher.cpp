// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "lib_ops/api/operator_dispatcher.h"
#include "lib_ops/api/par.h"

namespace xcore {

constexpr size_t bytes_per_stackword = 4;

OperatorDispatcher& GetOperatorDispatcher() {
  static OperatorDispatcher dispatcher;
  return dispatcher;
}

XCoreStatus InitializeDispatcher(void* buffer, size_t size) {
  OperatorDispatcher& dispatcher = GetOperatorDispatcher();
  return dispatcher.SetAllocatorBuffer(buffer, size);
}

#ifdef XCORE
// xCORE OperatorDispatcher implementation.
// Uses a threadgroup_t to dispatch kernel funnctions to HW threads.
OperatorDispatcher::OperatorDispatcher(bool use_current)
    : use_current_(use_current), reserved_stack_(0), stack_ptr_(nullptr) {
  group_ = thread_group_alloc();
}
OperatorDispatcher::~OperatorDispatcher() { thread_group_free(group_); }

void OperatorDispatcher::Start() {
  int begin = 0;

  if (use_current_) {
    const KernelCommand& command = commands_.data[begin];
    (command.function)(command.argument);
    begin++;
  }

  for (int i = begin; i < commands_.size; i++) {
    const KernelCommand& command = commands_.data[i];
    thread_group_add(group_, command.function, command.argument,
                     stack_base(command.stack, command.stack_words));
  }

  thread_group_start(group_);
}

void OperatorDispatcher::Wait() { thread_group_wait(group_); }

#else
// x86 OperatorDispatcher implementation.
// Uses a std::vector of std::thread to dispatch kernel funnctions to SW
// threads.
OperatorDispatcher::OperatorDispatcher(bool use_current)
    : use_current_(use_current), stack_ptr_(nullptr) {
  commands_.size = 0;
}

OperatorDispatcher::~OperatorDispatcher() {}

void OperatorDispatcher::Start() {
  int begin = 0;
  if (use_current_) {
    const KernelCommand& command = commands_.data[begin];
    (command.function)(command.argument);
    begin++;
  }

  for (int i = begin; i < commands_.size; i++) {
    const KernelCommand& command = commands_.data[i];
    group_.push_back(std::thread(command.function, command.argument));
  }
}

void OperatorDispatcher::Wait() {
  for (auto& thread : group_) {
    thread.join();
  }
  group_.clear();

  commands_.size = 0;
}

#endif

XCoreStatus OperatorDispatcher::SetAllocatorBuffer(void* buffer, size_t size) {
  allocator_.SetBuffer(buffer, size);

  return kXCoreOk;
}

XCoreStatus OperatorDispatcher::Reset() {
  reserved_stack_ = 0;
  stack_ptr_ = nullptr;
  allocator_.Reset();

  return kXCoreOk;
}

void* OperatorDispatcher::AllocateStackBuffer(int32_t num_threads,
                                              size_t stack_words) {
  assert(num_threads <= maxthreads);
  size_t stack_size = stack_words * bytes_per_stackword * num_threads;
  reserved_stack_ = std::max(stack_size, reserved_stack_);
  // reserved_threads_ = std::max(num_threads, reserved_threads_);
  stack_ptr_ = reinterpret_cast<char*>(
      allocator_.Reallocate(stack_ptr_, reserved_stack_));

  return stack_ptr_;
}

void* OperatorDispatcher::AllocatePersistentBuffer(size_t size) {
  return allocator_.Allocate(size);
}

XCoreStatus OperatorDispatcher::Add(kernel_function_t function, void* argument,
                                    size_t stack_words) {
  assert(stack_ptr_);
  assert(commands_.size < maxthreads);

  int32_t offset = stack_words * bytes_per_stackword * commands_.size;

  if (offset > reserved_stack_) {
    return kXCoreError;
  }

  void* stack = stack_ptr_ + offset;
  commands_.data[commands_.size] = {function, (void*)argument, stack_words,
                                    stack};
  commands_.size++;

  return kXCoreOk;
}

}  // namespace xcore
