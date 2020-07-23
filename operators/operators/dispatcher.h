// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_DISPATCHER_H_
#define XCORE_OPERATORS_DISPATCHER_H_

#include "operators/planning.h"
#include "operators/xcore_reporter.h"
#include "tensorflow/lite/c/common.h"

#ifdef XCORE
extern "C" {
#ifdef _TIME_H_
#define _clock_defined
#endif
#include <xcore/thread.h>
}

#define ATTRIBUTE_THREAD_FUNCTION __attribute__((fptrgroup("thread_function")))
#define STRINGIFY(NAME) #NAME
// #define GET_STACKWORDS(DEST, NAME) \
//   asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))
#define GET_STACKSIZE(DEST, NAME)                        \
  {                                                      \
    size_t _stack_words;                                 \
    asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" \
        : [ __dest ] "=r"(_stack_words));                \
    DEST = (_stack_words + 2) * 4;                       \
  }
#define IS_RAM(a) (((uintptr_t)a >= 0x80000) && ((uintptr_t)a <= 0x100000))
#define IS_NOT_RAM(a) ((uintptr_t)a > 0x100000)

#else  // not XCORE
#include <thread>
#include <vector>

#define ATTRIBUTE_THREAD_FUNCTION
#define GET_STACKSIZE(DEST, NAME) DEST = 0
#define IS_RAM(a) (1)
#define IS_NOT_RAM(a) (0)

typedef void (*thread_function_t)(void *);
typedef std::vector<std::thread> threadgroup_t;
#endif

namespace xcore {

constexpr size_t kMaxThreads = 5;
constexpr size_t kBytesPerStackword = 4;
constexpr size_t kWordAlignment = 4;
constexpr size_t kDoubleWordAlignment = 8;

typedef struct TaskArray {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  size_t stack_size;
  char *stack;
  int size;
  void *arguments[kMaxThreads];
} TaskArray;

class Dispatcher {
 public:
  Dispatcher(tflite::ErrorReporter *reporter, bool use_current_core = true);
  ~Dispatcher();

  TfLiteStatus InitializeTasks(thread_function_t function, char *stack,
                               size_t stack_size);
  TfLiteStatus AddTask(void *argument);
  TfLiteStatus JoinTasks();

  TfLiteStatus Reset();

  tflite::ErrorReporter *GetReporter();

  void FetchBuffer(int8_t **dest, int8_t const *src, size_t size);
  void FetchWeights(int8_t **dest, int8_t const *src, size_t size,
                    ChannelGroup const &changrp);
  void FetchBiases(int16_t **dest, int16_t const *src, size_t size,
                   ChannelGroup const &changrp);

 private:
  bool use_current_thread_;
  threadgroup_t group_;
  TaskArray tasks_;
  tflite::ErrorReporter *reporter_;
};

// static, shared Dispatcher object
Dispatcher *GetDispatcher();
void SetDispatcher(Dispatcher *);

}  // namespace xcore

#endif  // XCORE_OPERATORS_DISPATCHER_H_
