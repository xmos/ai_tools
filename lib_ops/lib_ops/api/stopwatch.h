// Copyright (c) 2020, XMOS Ltd, All rights reserved

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
