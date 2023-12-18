// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_THREADSUPPORT_H
#define XFORMER_UTILS_THREADSUPPORT_H

#include "llvm/ADT/SmallVector.h"

#include <array>

namespace mlir {
namespace xcore {
namespace utils {

llvm::SmallVector<std::array<int, 4>>
getImageRegionThreadSplits(const int &threadCount, const int &imageHeight,
                           const int &imageWidth, const int subH = 0,
                           const int subW = 0, const int strideH = 1,
                           const int strideW = 1);

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_THREADSUPPORT_H
