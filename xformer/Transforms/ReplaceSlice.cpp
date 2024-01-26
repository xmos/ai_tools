// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace TFL Slice with Slice for XCore.
struct ReplaceSlice
    : public PassWrapper<ReplaceSlice, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceSlice)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-slice"; }
  StringRef getDescription() const final {
    return "Replace TFL Slice with Slice for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceSlicePattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    auto inputType = sliceOp.getInput().getType().cast<RankedTensorType>();
    auto inputElementType = inputType.getElementType();

    // Check for invalid types and return
    // Input type must be QI8
    if (!(inputElementType.isa<quant::QuantizedType>() &&
          inputElementType.cast<quant::QuantizedType>().isSigned() &&
          inputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    auto outputType = sliceOp.getOutput().getType().cast<RankedTensorType>();
    auto outputElementType = outputType.getElementType();

    // Output type must be QI8
    if (!(outputElementType.isa<quant::QuantizedType>() &&
          outputElementType.cast<quant::QuantizedType>().isSigned() &&
          outputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Check if both input and output tensors have a rank of 4
    if (inputType.getRank() != 4 || outputType.getRank() != 4) {
      return failure();
    }

    DenseElementsAttr beginAttr;
    matchPattern(sliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr sizeAttr;
    matchPattern(sliceOp.getSize(), m_Constant(&sizeAttr));
    auto sizeValues = sizeAttr.getValues<int32_t>();

    if (beginValues[3] != 0) {
      return failure();
    }
    for (int i = 0; i < 3; i++) {
      if (beginValues[i] != 0 || sizeValues[i] != inputType.getDimSize(i)) {
        return failure();
      }
    }

    SliceMemcpyType memcpyType;
    if (inputType.getDimSize(2) == outputType.getDimSize(2) &&
        inputType.getDimSize(3) == outputType.getDimSize(3)) {
      // Single CPU memcpy
      memcpyType = SliceMemcpyType::SliceCpy;
    } else if (inputType.getDimSize(3) % 4 == 0 &&
               outputType.getDimSize(3) % 4 == 0) {
      // If not a slice copy, then if both depths are multiples of four, we can
      // do pixel by pixel VPU copy.
      memcpyType = SliceMemcpyType::VpuCpy;
    } else {
      // Pixel by pixel CPU memcpy, when depth not a multiple of four
      memcpyType = SliceMemcpyType::MemCpy;
    }

    int32_t inputHeight = inputType.getDimSize(1);
    int32_t inputWidth = inputType.getDimSize(2);
    int32_t inputDepth = inputType.getDimSize(3);
    int32_t beginX = beginValues[2];
    int32_t beginY = beginValues[1];

    auto image_geom = nn::ImageGeometry(inputHeight, inputWidth, inputDepth);
    auto window_geom =
        nn::WindowGeometry({sizeValues[2], sizeValues[1], inputDepth},
                           {beginY, beginX}, {1, 1, 1}, {1, 1});

    nn::ImToColValid imToCol(image_geom, window_geom,
                             static_cast<int>(inputDepth),
                             /*dont_zero_pad_at_the_end=*/true);
    auto imToColParams = imToCol.getParams();
    auto mfStr = std::string((char *)&imToColParams, sizeof(imToColParams));

    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), sliceOp.getInput(),
        rewriter.getI32IntegerAttr(beginX), rewriter.getI32IntegerAttr(beginY),
        rewriter.getStringAttr(mfStr),
        rewriter.getI32IntegerAttr((int32_t)memcpyType));
    rewriter.replaceOp(sliceOp, binaryObjectSliceOp.getOutput());

    return success();
  }
};

void ReplaceSlice::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceSlice pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSlicePass() {
  return std::make_unique<ReplaceSlice>();
}

static PassRegistration<ReplaceSlice> pass;

} // namespace xcore
} // namespace mlir
