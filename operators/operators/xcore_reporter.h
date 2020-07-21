// Copyright (c) 2019, XMOS Ltd, All rights reserved
#ifndef XCORE_REPORTER_H_
#define XCORE_REPORTER_H_

#include <cstdarg>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/compatibility.h"

namespace xcore {

class XCoreReporter : public tflite::ErrorReporter {
 public:
  ~XCoreReporter() override {}
  int Report(const char* format, va_list args) override;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_REPORT_STATUS(...) TF_LITE_REPORT_ERROR(__VA_ARGS__)
#else  // TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_REPORT_STATUS(...)
#endif  // TF_LITE_STRIP_ERROR_STRINGS

#endif  // XCORE_REPORTER_H_