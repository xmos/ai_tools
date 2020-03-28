// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "lib_ops/api/operator_dispatcher.h"

namespace xcore {

constexpr size_t maxthreads = 8;
constexpr size_t bytes_per_stackword = 4;

OperatorDispatcher& GetOperatorDispatcher() {
  static OperatorDispatcher dispatcher;
  return dispatcher;
}

XCoreStatus AllocateOperatorDispatcher(/*TODO: allocator here*/) {
  OperatorDispatcher& dispatcher = GetOperatorDispatcher();
  return dispatcher.Allocate();
}

struct KernelCommand {
  ATTRIBUTE_KERNEL_FUNCTION kernel_function_t function;
  void* argument;
  size_t stack_words;
  void* stack;
  KernelCommand(kernel_function_t f, void* a = nullptr, size_t w = 0,
                void* s = nullptr)
      : function(f), argument(a), stack_words(w), stack(s) {}
};

#ifdef XCORE
// xCORE OperatorDispatcher implementation.
// Uses a threadgroup_t to dispatch kernel funnctions to HW threads.
OperatorDispatcher::OperatorDispatcher(bool use_current)
    : use_current_(use_current), stack_ptr_(nullptr) {
  group_ = thread_group_alloc();
}
OperatorDispatcher::~OperatorDispatcher() { thread_group_free(group_); }

void OperatorDispatcher::Start() {
  auto cend = commands_.cend();

  if (use_current_) --cend;

  for (auto it = commands_.cbegin(); it < cend; ++it) {
    thread_group_add(group_, it->function, it->argument,
                     stack_base(it->stack, it->stack_words));
  }
  thread_group_start(group_);

  if (use_current_) {
    KernelCommand const& command = commands_.back();
    (*command.function)(command.argument);
  }
}

void OperatorDispatcher::Wait() { thread_group_wait(group_); }

#else
// x86 OperatorDispatcher implementation.
// Uses a std::vector of std::thread to dispatch kernel funnctions to SW
// threads.
OperatorDispatcher::OperatorDispatcher(bool use_current)
    : use_current_(use_current), stack_ptr_(nullptr) {}
OperatorDispatcher::~OperatorDispatcher() {}

void OperatorDispatcher::Start() {
  auto cbegin = commands_.cbegin();

  if (use_current_) {
    KernelCommand const& command = *cbegin;
    (*command.function)(command.argument);
    ++cbegin;
  }

  for (auto it = cbegin; it < commands_.cend(); ++it) {
    group_.push_back(std::thread(it->function, it->argument));
  }


}

void OperatorDispatcher::Wait() {
  for (auto& thread : group_) {
    thread.join();
  }

  group_.clear();
  commands_.clear();
}

#endif

void OperatorDispatcher::Reserve(int32_t num_threads, size_t stack_words) {
  assert(num_threads <= maxthreads);
  size_t stack_size = stack_words * bytes_per_stackword * num_threads;
  reserved_stack_ = std::max(stack_size, reserved_stack_);
  reserved_threads_ = std::max(num_threads, reserved_threads_);
}

XCoreStatus OperatorDispatcher::Allocate(/*TODO: allocator here*/) {
  stack_ptr_ = (char*)malloc(reserved_stack_);

  if (stack_ptr_) return kXCoreOk;

  return kXCoreError;
}

XCoreStatus OperatorDispatcher::Add(kernel_function_t function, void* argument,
                                  size_t stack_words) {
  assert(stack_ptr_);
  int32_t offset = stack_words * bytes_per_stackword * commands_.size();

  if (offset > reserved_stack_) {
    return kXCoreError;
  }

  void* stack = stack_ptr_ + offset;
  commands_.emplace_back(function, (void*)argument, stack_words, stack);
  assert(commands_.size() <= maxthreads);

  return kXCoreOk;
}

}  // namespace xcore
