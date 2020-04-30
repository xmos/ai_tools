// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef LIB_OPS_STOPWATCH_H_
#define LIB_OPS_STOPWATCH_H_

#ifdef XCORE
extern "C" {
#include "lib_ops/src/xs1.h"  // FIXME: remove someday
//    this must appear BEFORE including xcore/hwtimer.h
#include <xcore/hwtimer.h>
}
#else  // not XCORE
#include <chrono>
#endif  // XCORE

namespace xcore {

class Stopwatch {
 public:
  Stopwatch();
  ~Stopwatch();
  void Start();
  void Stop();
  int GetEllapsedNanoseconds();
  int GetEllapsedMicroseconds();

 private:
#ifdef XCORE
  hwtimer_t hwtimer;
  unsigned start_;
  unsigned stop_;
#else   // not XCORE
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> stop_;
#endif  // XCORE
};
}  // namespace xcore

//*****************************
//*****************************
//*****************************
// Macros for benchmarking
//*****************************
//*****************************
//*****************************

#ifdef ENABLE_BENCHMARKING

#define TIMER_START()      \
  xcore::Stopwatch __sw__; \
  __sw__.Start()

#define TIMER_STOP(...) \
  __sw__.Stop();        \
  printf(__VA_ARGS__);  \
  printf(" : %u (us)\n", __sw__.GetEllapsedMicroseconds())
#else

#define TIMER_START()
#define TIMER_STOP(...)

#endif  // ENABLE_BENCHMARKING

#endif  // LIB_OPS_STOPWATCH_H_