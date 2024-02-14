// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}

namespace mlir {
namespace xcore {

namespace {

size_t getTypeSize(Type type) {
  if (auto quantType = type.dyn_cast<UniformQuantizedType>()) {
    return quantType.getStorageType().getIntOrFloatBitWidth() / 8;
  } else if (auto floatType = type.dyn_cast<FloatType>()) {
    return floatType.getWidth() / 8;
  } else if (auto intType = type.dyn_cast<IntegerType>()) {
    return intType.getWidth() / 8;
  } else {
    llvm_unreachable("Unsupported type");
  }
  return 0;
}

// Replace TFL Pad with Pad for XCore.
struct ReplacePad
    : public PassWrapper<ReplacePad, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplacePad)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-pad"; }
  StringRef getDescription() const final {
    return "Replace TFL Pad with Pad for XCore.";
  }
  void runOnOperation() override;
};

struct ReplacePadPattern : public OpRewritePattern<TFL::PadOp> {
  using OpRewritePattern<TFL::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::PadOp padOp,
                                PatternRewriter &rewriter) const override {

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the pad into a constant later.
    if (matchPattern(padOp.getInput(), m_Constant()) ||
        matchPattern(padOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    auto inputType = padOp.getInput().getType().cast<RankedTensorType>();
    if (!inputType.hasStaticShape()) {
      return failure();
    }
    Type inputElementType = inputType.getElementType();
    auto outputType = padOp.getOutput().getType().cast<RankedTensorType>();

    DenseElementsAttr paddingAttr;
    matchPattern(padOp.getPadding(), m_Constant(&paddingAttr));
    auto paddingValues = paddingAttr.getValues<int32_t>();

    const int rank = inputType.getRank();

    int beginValues[rank], sizeValues[rank];
    for (int i = 0; i < rank; i++) {
      beginValues[i] = paddingValues[i * 2];
      sizeValues[i] = inputType.getShape()[i];
    }
    // Check if the pad is a no-op
    bool isNoOp = true;
    for (int i = 0; i < rank; i++) {
      if (beginValues[i] != 0 || sizeValues[i] != outputType.getShape()[i]) {
        isNoOp = false;
        break;
      }
    }
    if (isNoOp) {
      rewriter.replaceOp(padOp, padOp.getInput());
      return success();
    }

    int begin_dst[5], end_dst[5], in_offsets[4], out_offsets[4], shape_dst[5];

    // TFLite supports up to 5 dimensions, if the input is less we pad
    const size_t dtype_size = getTypeSize(inputElementType);

    // Cast beginValues and sizeValues to int* for slice_memcpy_get_params
    int begin[5], size[5];
    for (int i = 0; i < rank; i++) {
      begin[i] = beginValues[i];
      size[i] = sizeValues[i];
    }

    int shape[5];
    for (int i = 0; i < rank; i++) {
      shape[i] = outputType.getShape()[i];
    }
    slice_memcpy_get_params(begin_dst, end_dst, in_offsets, out_offsets,
                            shape_dst, begin, size, shape, dtype_size, rank);
    const bool isVpu =
        shape_dst[4] % 4 == 0 && begin_dst[4] % 4 == 0 && end_dst[4] % 4 == 0;

    auto binaryObjectPadOp = rewriter.create<PadOp>(
        padOp.getLoc(), padOp.getType(), padOp.getInput(),
        rewriter.getI32ArrayAttr(begin_dst), rewriter.getI32ArrayAttr(end_dst),
        rewriter.getI32ArrayAttr(in_offsets),
        rewriter.getI32ArrayAttr(out_offsets), rewriter.getBoolAttr(isVpu));

    rewriter.replaceOp(padOp, binaryObjectPadOp.getOutput());

    return success();
  }
};

void ReplacePad::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplacePadPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplacePad pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplacePadPass() {
  return std::make_unique<ReplacePad>();
}

static PassRegistration<ReplacePad> pass;

} // namespace xcore
} // namespace mlir
