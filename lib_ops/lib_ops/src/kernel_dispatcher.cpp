// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <cassert>

#include "lib_ops/api/kernel_dispatcher.h"

namespace xcore {

constexpr size_t maxthreads = 8;
constexpr size_t bytes_per_stackword = 4;

KernelDispatcher& GetKernelDispatcher() {
  static KernelDispatcher dispatcher;
  return dispatcher;
}

XCoreStatus AllocateKernelDispatcher(/*TODO: allocator here*/) {
  KernelDispatcher& dispatcher = GetKernelDispatcher();
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
// xCORE KernelDispatcher implementation.
// Uses a threadgroup_t to dispatch kernel funnctions to HW threads.
KernelDispatcher::KernelDispatcher(bool use_current)
    : use_current_(use_current), stack_ptr_(nullptr) {
  group_ = thread_group_alloc();
}
KernelDispatcher::~KernelDispatcher() { thread_group_free(group_); }

void KernelDispatcher::Start() {
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

void KernelDispatcher::Wait() { thread_group_wait(group_); }

#else
// x86 KernelDispatcher implementation.
// Uses a std::vector of std::thread to dispatch kernel funnctions to SW
// threads.
KernelDispatcher::KernelDispatcher(bool use_current)
    : use_current_(use_current), stack_ptr_(nullptr) {}
KernelDispatcher::~KernelDispatcher() {}

void KernelDispatcher::Start() {
  auto cend = commands_.cend();

  if (use_current_) --cend;

  for (auto it = commands_.cbegin(); it < cend; ++it) {
    group_.push_back(std::thread(it->function, it->argument));
  }

  if (use_current_) {
    KernelCommand const& command = commands_.back();
    (*command.function)(command.argument);
  }
}

void KernelDispatcher::Wait() {
  for (auto& thread : group_) {
    thread.join();
  }

  group_.clear();
}

#endif

void KernelDispatcher::Reserve(int32_t num_threads, size_t stack_words) {
  assert(num_threads <= maxthreads);
  size_t stack_size = stack_words * bytes_per_stackword * num_threads;
  reserved_stack_ = std::max(stack_size, reserved_stack_);
  reserved_threads_ = std::max(num_threads, reserved_threads_);
}

XCoreStatus KernelDispatcher::Allocate(/*TODO: allocator here*/) {
  stack_ptr_ = (char*)malloc(reserved_stack_);

  if (stack_ptr_) return kXCoreOk;

  return kXCoreError;
}

XCoreStatus KernelDispatcher::Add(kernel_function_t function, void* argument,
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
