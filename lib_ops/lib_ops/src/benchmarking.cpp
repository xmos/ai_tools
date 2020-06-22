// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/benchmarking.h"

namespace xcore {

#ifdef XCORE

Stopwatch::Stopwatch() : start_(0), stop_(0) {}
Stopwatch::~Stopwatch() {}
void Stopwatch::Start() {
  hwtimer_ = hwtimer_alloc();
  start_ = hwtimer_get_time(hwtimer_);
}
void Stopwatch::Stop() {
  stop_ = hwtimer_get_time(hwtimer_);
  hwtimer_free(hwtimer_);
}
int Stopwatch::GetEllapsedNanoseconds() { return (stop_ - start_) * 10; }
int Stopwatch::GetEllapsedMicroseconds() { return (stop_ - start_) / 100; }

#else  // not XCORE

Stopwatch::Stopwatch() {}
Stopwatch::~Stopwatch() {}
void Stopwatch::Start() { start_ = std::chrono::steady_clock::now(); }
void Stopwatch::Stop() { stop_ = std::chrono::steady_clock::now(); }
int Stopwatch::GetEllapsedNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_ - start_)
      .count();
}
int Stopwatch::GetEllapsedMicroseconds() {
  return std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_)
      .count();
}

#endif  // XCORE

}  // namespace xcore
