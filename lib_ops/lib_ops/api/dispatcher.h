// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_DISPATCHER_H_
#define XCORE_OPERATORS_DISPATCHER_H_

#include "lib_ops/api/allocator.h"
#include "lib_ops/api/lib_ops.h"
#include "lib_ops/api/planning.h"

#ifdef XCORE

extern "C" {
#include "lib_ops/src/xs1.h"  // FIXME: remove someday
//    this must appear BEFORE including xcore/thread.h
#include <xcore/thread.h>
}

#define ATTRIBUTE_THREAD_FUNCTION __attribute__((fptrgroup("thread_function")))
#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))

#else  // not XCORE
#include <thread>
#include <vector>

#define ATTRIBUTE_THREAD_FUNCTION
#define GET_STACKWORDS(DEST, NAME) DEST = 0

typedef void (*thread_function_t)(void *);
typedef std::vector<std::thread> threadgroup_t;
#endif

namespace xcore {

typedef struct Task {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  void *argument;
} Task;

typedef struct TaskArray {
  int size;
  size_t stack_words;
  Task *data;
} TaskArray;

class Dispatcher {
 public:
  Dispatcher(void *buffer, size_t size, int num_cores,
             bool use_current_core = true);
  ~Dispatcher();

  XCoreStatus AddThread(thread_function_t function, void *argument,
                        size_t stack_words);
  XCoreStatus Join();
  XCoreStatus Reset();
  XCoreStatus ResetScratchAllocation();

 private:
  int num_threads_;
  bool use_current_thread_;
  threadgroup_t group_;
  TaskArray tasks_;
  MemoryAllocator allocator_;
};

// static, shared Dispatcher object
Dispatcher *GetDispatcher();
XCoreStatus InitializeXCore(Dispatcher *dispatcher);

}  // namespace xcore

#endif  // XCORE_OPERATORS_DISPATCHER_H_
