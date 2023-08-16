// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/ThreadSupport.h"

#include <cmath>

namespace mlir {
namespace xcore {
namespace utils {

llvm::SmallVector<std::array<int, 4>>
getImageRegionThreadSplits(const int &numThreads, const int &imageHeight,
                           const int &imageWidth) {
  // Decide between height or width as the chosen dimension to split
  //
  // If we can cleanly divide a dimension by numThreads, we choose that one
  // If that's not possible, we pick the larger dimension
  int dimSize;
  bool isHeightChosenDim;
  if (imageHeight % numThreads == 0) {
    dimSize = imageHeight;
    isHeightChosenDim = true;
  } else if (imageWidth % numThreads == 0) {
    dimSize = imageWidth;
    isHeightChosenDim = false;
  } else {
    dimSize = imageHeight > imageWidth ? imageHeight : imageWidth;
    isHeightChosenDim = imageHeight > imageWidth ? true : false;
  }

  // Divide the work among the threads
  //
  // For e.g.,
  // If the dim size is 6 and numThreads is 4,
  // we split the work as 2, 2, 1, 1
  //
  // If the dim size is smaller than numThreads,
  // For e.g.,
  // If the dim size is 2 and numThreads is 3,
  // we will only use two threads
  llvm::SmallVector<int> dimSplits;
  for (int i = numThreads; i > 0; --i) {
    auto split = static_cast<int>(ceil(double(dimSize) / double(i)));
    dimSize -= split;
    if (split > 0) {
      dimSplits.push_back(split);
    }
  }

  // Create imageRegions for each of the threads
  // An imageregion is of the form
  // {topLeftRow, topLeftColumn, numberOfRows, numberOfColumns}
  int topLeftRow = 0;
  int topLeftColumn = 0;
  llvm::SmallVector<std::array<int, 4>> imageRegionSplits;
  for (auto &split : dimSplits) {
    if (isHeightChosenDim) {
      imageRegionSplits.push_back(
          {topLeftRow, topLeftColumn, split, imageWidth});
      topLeftRow += split;
    } else {
      imageRegionSplits.push_back(
          {topLeftRow, topLeftColumn, imageHeight, split});
      topLeftColumn += split;
    }
  }
  return imageRegionSplits;
}

} // namespace utils
} // namespace xcore
} // namespace mlir
