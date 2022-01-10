// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/ConvPatterns.h"

namespace mlir {
namespace xcore {

// Handle Larq BNN Conv2D
LogicalResult ReplaceBConv2DPattern::checkIfValid(lq::Bconv2dOp op) const {
  llvm::dbgs() << "bconv2d check\n";
  return success();
}

LogicalResult ReplaceBConv2DPattern::getKernelType(const BConvArgs &args,
                                                   Conv2DType &kt) const {
  llvm::dbgs() << "bconv2d getkerneltype\n";
  return success();
}

LogicalResult ReplaceBConv2DPattern::getArgs(lq::Bconv2dOp op,
                                             BConvArgs &args) const {
  return success();
}

LogicalResult ReplaceBConv2DPattern::getSerializedParamsAndTensors(
    const BConvArgs &args, const Conv2DType &kt,
    llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsTensorData,
    std::vector<int16_t> &multipliersAndBiasesTensorData,
    int &scratchBytes) const {
  llvm::dbgs() << "bconv2d getSerializedParamsAndTensors\n";

  return success();
}

} // namespace xcore
} // namespace mlir
