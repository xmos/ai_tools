// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATOR_DISPATCHER_H_
#define XCORE_OPERATOR_DISPATCHER_H_

#include "lib_ops/api/par.h"
#include "lib_ops/api/operator_allocator.h"
#include "lib_ops/api/lib_ops.h"

#ifdef XCORE

extern "C" {
#include "lib_ops/src/xs1.h"  // FIXME: remove someday
//    this must appear BEFORE including xcore/thread.h
#include <xcore/thread.h>
}

#define ATTRIBUTE_KERNEL_FUNCTION __attribute__((fptrgroup("kernel_function")))
#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))

typedef thread_function_t kernel_function_t;
typedef threadgroup_t thread_group_t;
#else // not XCORE
#include <vector>
#include <thread>

#define ATTRIBUTE_KERNEL_FUNCTION
#define GET_STACKWORDS(DEST, NAME) DEST=0

typedef void (*kernel_function_t)(void*);
typedef std::vector<std::thread> thread_group_t;
#endif

namespace xcore {

typedef struct KernelCommand {
  ATTRIBUTE_KERNEL_FUNCTION kernel_function_t function;
  void* argument;
  size_t stack_words;
  void* stack;
} KernelCommand;

typedef struct KernelCommandArray {
  int size;
  KernelCommand data[maxthreads];
} KernelCommandArray;

class OperatorDispatcher {
 public:
  OperatorDispatcher(bool use_current = true);
  ~OperatorDispatcher();

  void* AllocateStackBuffer(int32_t num_threads, size_t stack_words);
  void* AllocatePersistentBuffer(size_t size);
  XCoreStatus SetAllocatorBuffer(void* buffer, size_t size);
  XCoreStatus Reset();
  XCoreStatus Add(kernel_function_t function, void* argument,
                  size_t stack_words);
  void Start();
  void Wait();

 private:
  bool use_current_;
  //int32_t reserved_threads_;
  size_t reserved_stack_;
  char* stack_ptr_;
  thread_group_t group_;
  KernelCommandArray commands_;
  LinearAllocator allocator_;
};

// static, shared OperatorDispatcher object
OperatorDispatcher& GetOperatorDispatcher();
XCoreStatus InitializeDispatcher(void* buffer, size_t size);

}  // namespace xcore

#endif  // XCORE_OPERATOR_DISPATCHER_H_
