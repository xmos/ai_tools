// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/ThreadSupport.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/PriorityQueue.h"

#include <cmath>

namespace mlir {
namespace xcore {
namespace utils {

namespace {

mlir::LogicalResult
getSmallDimSplits(llvm::SmallVector<std::array<int, 4>> &imageRegionSplits,
                  const int &numThreads, const int &dimSize,
                  bool isHeightChosenDim, const int &alignedImageHeight,
                  const int &alignedImageWidth, const int &imageHeight,
                  const int &imageWidth, const int &subH, const int &subW,
                  const int &strideH, const int &strideW) {
  if (dimSize >= numThreads) {
    return failure();
  }

  int otherDim;
  llvm::SmallVector<int, 5> x, y, h, w, xx, yy, hh, ww;
  if (isHeightChosenDim) {
    otherDim = alignedImageWidth;
  } else {
    otherDim = alignedImageHeight;
  }
  if (otherDim == 1) {
    return failure();
  }

  if (dimSize == 4) {
    assert(numThreads == 5);
    if (otherDim == 4) {
      xx = {0, 1, 2, 3, 3};
      yy = {0, 0, 0, 0, 2};
      hh = {1, 1, 1, 1, 1};
      ww = {4, 4, 4, 2, 2};
    } else if (otherDim == 3) {
      xx = {0, 1, 2, 3, 3};
      yy = {0, 0, 0, 0, 2};
      hh = {1, 1, 1, 1, 1};
      ww = {3, 3, 3, 2, 1};
    } else if (otherDim == 2) {
      xx = {0, 1, 2, 3, 3};
      yy = {0, 0, 0, 0, 1};
      hh = {1, 1, 1, 1, 1};
      ww = {2, 2, 2, 1, 1};
    }
  } else if (dimSize == 3) {
    if (numThreads == 4) {
      if (otherDim == 3) {
        xx = {0, 1, 2, 2};
        yy = {0, 0, 0, 2};
        hh = {1, 1, 1, 1};
        ww = {3, 3, 2, 1};
      } else if (otherDim == 2) {
        xx = {0, 1, 2, 2};
        yy = {0, 0, 0, 1};
        hh = {1, 1, 1, 1};
        ww = {2, 2, 1, 1};
      }
    } else {
      assert(numThreads == 5);
      if (otherDim == 3) {
        xx = {0, 1, 1, 2, 2};
        yy = {0, 0, 2, 0, 2};
        hh = {1, 1, 1, 1, 1};
        ww = {3, 2, 1, 2, 1};
      } else if (otherDim == 2) {
        xx = {0, 1, 1, 2, 2};
        yy = {0, 0, 1, 0, 1};
        hh = {1, 1, 1, 1, 1};
        ww = {2, 1, 1, 1, 1};
      }
    }

  } else if (dimSize == 2) {
    if (numThreads == 3) {
      assert(otherDim == 2);
      xx = {0, 1, 1};
      yy = {0, 0, 1};
      hh = {1, 1, 1};
      ww = {2, 1, 1};
    } else {
      assert(numThreads == 4 || numThreads == 5);
      assert(otherDim == 2);
      xx = {0, 0, 1, 1};
      yy = {0, 1, 0, 1};
      hh = {1, 1, 1, 1};
      ww = {1, 1, 1, 1};
    }
  }

  if (isHeightChosenDim) {
    x = xx;
    y = yy;
    h = hh;
    w = ww;
  } else {
    x = yy;
    y = xx;
    h = ww;
    w = hh;
  }

  for (int i = 0; i < x.size(); i++) {
    x[i] = x[i] * strideH;
    y[i] = y[i] * strideW;
    h[i] = h[i] * strideH;
    w[i] = w[i] * strideW;
  }

  // If odd sizes, will need to adjust the last split
  if (isHeightChosenDim && imageHeight % 2 == 1) {
    h[h.size() - 1]++;
  } else if (!isHeightChosenDim && imageWidth % 2 == 1) {
    w[w.size() - 1]++;
  }

  for (int i = 0; i < x.size(); i++) {
    if (isHeightChosenDim) {
      h[i] = (x[i] + subH + h[i] > imageHeight && imageHeight % 2 == 1)
                 ? h[i] - 1
                 : h[i];
    } else {
      w[i] = (y[i] + subW + w[i] > imageWidth && imageWidth % 2 == 1) ? w[i] - 1
                                                                      : w[i];
    }
    imageRegionSplits.push_back({x[i] + subH, y[i] + subW, h[i], w[i]});
  }
  return success();
}
} // namespace

llvm::SmallVector<std::array<int, 4>> getImageRegionThreadSplits(
    const int &numThreads, const int &imageHeight, const int &imageWidth,
    const int subH, const int subW, const int strideH, const int strideW) {
  // Decide between height or width as the chosen dimension to split
  //
  // If we can cleanly divide a dimension by numThreads, we choose that one
  // If that's not possible, we pick the larger dimension
  int dimSize;
  bool isHeightChosenDim;
  // When we multi-thread Transpose Conv, we have to take output strides into
  // account
  auto alignedImageHeight = imageHeight / strideH;
  auto alignedImageWidth = imageWidth / strideW;

  if (alignedImageHeight % numThreads == 0) {
    dimSize = alignedImageHeight;
    isHeightChosenDim = true;
  } else if (alignedImageWidth % numThreads == 0) {
    dimSize = alignedImageWidth;
    isHeightChosenDim = false;
  } else {
    dimSize = alignedImageHeight > alignedImageWidth ? alignedImageHeight
                                                     : alignedImageWidth;
    isHeightChosenDim = alignedImageHeight > alignedImageWidth ? true : false;
  }

  llvm::SmallVector<std::array<int, 4>> imageRegionSplits;
  // Handle small dim cases using tables
  // TODO
  // if (dimSize <= 4) {
  //   if (succeeded(getSmallDimSplits(imageRegionSplits, numThreads, dimSize,
  //                                   isHeightChosenDim, alignedImageHeight,
  //                                   alignedImageWidth, imageHeight,
  //                                   imageWidth, subH, subW, strideH,
  //                                   strideW))) {
  //     return imageRegionSplits;
  //   }
  // }

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
      if (isHeightChosenDim) {
        dimSplits.push_back(split * strideH);
      } else {
        dimSplits.push_back(split * strideW);
      }
    }
  }

  // If odd sizes, will need to adjust the last split
  if ((isHeightChosenDim && imageHeight % 2 == 1) ||
      (!isHeightChosenDim && imageWidth % 2 == 1)) {
    dimSplits[dimSplits.size() - 1]++;
  }

  // Create imageRegions for each of the threads
  // An imageregion is of the form
  // {topLeftRow, topLeftColumn, numberOfRows, numberOfColumns}
  int topLeftRow = 0;
  int topLeftColumn = 0;
  for (auto &split : dimSplits) {
    if (isHeightChosenDim) {
      split = (topLeftRow + subH + split > imageHeight && imageHeight % 2 == 1)
                  ? split - 1
                  : split;
      imageRegionSplits.push_back(
          {topLeftRow + subH, topLeftColumn, split, imageWidth});
      topLeftRow += split;
    } else {
      split = (topLeftColumn + subW + split > imageWidth && imageWidth % 2 == 1)
                  ? split - 1
                  : split;
      imageRegionSplits.push_back(
          {topLeftRow, topLeftColumn + subW, imageHeight, split});
      topLeftColumn += split;
    }
  }

  return imageRegionSplits;
}

} // namespace utils
} // namespace xcore
} // namespace mlir
