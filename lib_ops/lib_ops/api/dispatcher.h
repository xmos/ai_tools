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

constexpr size_t max_threads = 5;
constexpr size_t bytes_per_stackword = 4;
constexpr size_t changrp_len = (16);
constexpr size_t bso_changrp_len = (7 * changrp_len);
constexpr size_t bso_changrp_bytes = (bso_changrp_len * 2);

typedef struct TaskArray {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  size_t stack_words;
  char *stack;
  int size;
  void *arguments[max_threads];
} TaskArray;

class Dispatcher {
 public:
  Dispatcher(void *buffer, size_t buffer_size, bool use_current_core = true);
  ~Dispatcher();

  XCoreStatus InitializeTasks(thread_function_t function, size_t stack_words);
  XCoreStatus AddTask(void *argument);
  XCoreStatus JoinTasks();

  XCoreStatus Reset();

  void *AllocatePersistantBuffer(size_t size);
  void *AllocateScratchBuffer(size_t size);
  XCoreStatus ResetScratchAllocation();

  size_t GetMaxAllocatedSize();

  void PreloadBuffer(int8_t **dest, int8_t const *src, int32_t size);
  void PreloadWeights(int8_t **dest, int8_t const *src, int32_t size,
                      ChannelGroup const &changrp);
  void PreloadBiases(int16_t **dest, int16_t const *src,
                     ChannelGroup const &changrp);

 private:
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
