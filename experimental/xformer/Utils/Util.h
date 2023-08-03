// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_UTIL_H
#define XFORMER_UTILS_UTIL_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace xcore {
namespace utils {

int getShapedTypeSize(ShapedType t);

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_UTIL_H
