// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_CONVPATTERNS_H
#define XFORMER_TRANSFORMS_CONVPATTERNS_H

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "lib_nn/api/Conv2d.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

struct TFLConvArgs {
  int outputHeight, outputWidth, outputDepth, outputZeroPoint;
  int inputHeight, inputWidth, inputDepth, inputZeroPoint;
  int filterHeight, filterWidth, filterDepth;
  std::vector<int8_t> filter;
  std::vector<int32_t> bias;
  std::vector<float> effectiveMultiplier;
  int8_t padValue;
  bool toBePadded;
  nn::padding_t padding;
  nn::ImageGeometry Y;
  nn::ImageGeometry X;
  nn::WindowGeometry K;
};

struct BConvArgs {
  int outputHeight, outputWidth, outputDepth;
  int inputHeight, inputWidth, inputDepth;
  int filterHeight, filterWidth, filterDepth;
  std::vector<int32_t> filter;
  std::vector<float> bias;
  std::vector<float> effectiveMultiplier;
  std::vector<int32_t> threshold;
  int8_t padValue;
  nn::ImageGeometry Y;
  nn::ImageGeometry X;
  nn::WindowGeometry K;
};

//
// XC Conv2D Base class
// ConcreteType would be TFL Conv types or Larq BConv2D
// Replaces them with XC Conv2D
template <typename ConcreteType, typename ConvOpType, typename ArgsType>
class ReplaceWithXCConv2DBase : public OpRewritePattern<ConvOpType> {
public:
  ReplaceWithXCConv2DBase(MLIRContext *context)
      : OpRewritePattern<ConvOpType>(context) {}

  LogicalResult matchAndRewrite(ConvOpType op,
                                PatternRewriter &rewriter) const override;
};

//
//
//
// Handle Larq BNN Conv2D
class ReplaceBConv2DPattern
    : public ReplaceWithXCConv2DBase<ReplaceBConv2DPattern, lq::Bconv2dOp,
                                     BConvArgs> {
public:
  using BaseType =
      ReplaceWithXCConv2DBase<ReplaceBConv2DPattern, lq::Bconv2dOp, BConvArgs>;
  ReplaceBConv2DPattern(MLIRContext *context) : BaseType(context) {}

  LogicalResult checkIfValid(lq::Bconv2dOp op) const;

  LogicalResult getArgs(lq::Bconv2dOp op, BConvArgs &args) const;

  LogicalResult getKernelType(const BConvArgs &args, Conv2DType &kt) const;

  LogicalResult getSerializedParamsAndTensors(
      const BConvArgs &args, const Conv2DType &kt,
      llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;
};

//
//
// TFL Conv2D Base class
// ConcreteType would be TFL Conv2D or TFL DepthwiseConv2D
template <typename ConcreteType, typename TFLConvOpType>
class ReplaceConv2DBase : public ReplaceWithXCConv2DBase<
                              ReplaceConv2DBase<ConcreteType, TFLConvOpType>,
                              TFLConvOpType, TFLConvArgs> {
public:
  using BaseType =
      ReplaceWithXCConv2DBase<ReplaceConv2DBase<ConcreteType, TFLConvOpType>,
                              TFLConvOpType, TFLConvArgs>;
  ReplaceConv2DBase(MLIRContext *context) : BaseType(context) {}

  LogicalResult checkIfValid(TFLConvOpType op) const;

  LogicalResult getArgs(TFLConvOpType op, TFLConvArgs &args) const;

  LogicalResult getKernelType(const TFLConvArgs &args, Conv2DType &kt) const {
    if (failed(
            static_cast<const ConcreteType *>(this)->getKernelType(args, kt))) {
      return failure();
    }
    return success();
  }

  LogicalResult getSerializedParamsAndTensors(
      const TFLConvArgs &args, const Conv2DType &kt,
      llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const {
    if (failed(static_cast<const ConcreteType *>(this)
                   ->getSerializedParamsAndTensors(
                       args, kt, strParams, weightsTensorData,
                       multipliersAndBiasesTensorData, scratchBytes))) {
      return failure();
    }
    return success();
  }
};

// Handle TFL Conv2D
class ReplaceConv2DPattern
    : public ReplaceConv2DBase<ReplaceConv2DPattern, TFL::Conv2DOp> {
public:
  using BaseType = ReplaceConv2DBase<ReplaceConv2DPattern, TFL::Conv2DOp>;
  ReplaceConv2DPattern(MLIRContext *context) : BaseType(context) {}

  LogicalResult getKernelType(const TFLConvArgs &args, Conv2DType &kt) const;

  // Conv is quantized along dimension 0
  int getQuantizationIndex() const { return 0; }

  LogicalResult getSerializedParamsAndTensors(
      const TFLConvArgs &args, const Conv2DType &kt,
      llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;

private:
  LogicalResult getConv2DPaddedIndirectParams(
      const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;

  LogicalResult getConv2DValidIndirectParams(
      const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;

  LogicalResult getConv2DValidDirectParams(
      const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;
};

// Handle TFL DepthwiseConv2D
class ReplaceDepthwiseConv2DPattern
    : public ReplaceConv2DBase<ReplaceDepthwiseConv2DPattern,
                               TFL::DepthwiseConv2DOp> {
public:
  using BaseType =
      ReplaceConv2DBase<ReplaceDepthwiseConv2DPattern, TFL::DepthwiseConv2DOp>;
  ReplaceDepthwiseConv2DPattern(MLIRContext *context) : BaseType(context) {}

  LogicalResult getKernelType(const TFLConvArgs &args, Conv2DType &kt) const;

  // DepthwiseConv is quantized along dimension 3
  int getQuantizationIndex() const { return 3; }

  LogicalResult getSerializedParamsAndTensors(
      const TFLConvArgs &args, const Conv2DType &kt,
      llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;

private:
  LogicalResult getDepthwiseConv2DValidDirectParams(
      const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;

  LogicalResult getDepthwiseConv2DPaddedIndirectParams(
      const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
      std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData,
      int &scratchBytes) const;
};

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_CONVPATTERNS_H
