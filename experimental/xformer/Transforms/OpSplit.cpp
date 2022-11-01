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
// OpSplit
struct OpSplit
    : public PassWrapper<OpSplit,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSplit)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-opsplit"; }
  StringRef getDescription() const final {
    return "OpSplit.";
  }
  void runOnOperation() override;
};



struct OpSplitPattern
    : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  // LogicalResult matchAndRewrite(DummyStridedSliceOp stridedSliceOriginal,
  LogicalResult matchAndRewrite(TFL::Conv2DOp convOriginal,
                                PatternRewriter &rewriter) const override {


    if (!convOriginal.output().hasOneUse()) {
      return failure();
    }

    auto convReplacement   = rewriter.create<TFL::Conv2DOp>(
      convOriginal.getLoc(), convOriginal.getType(),  convOriginal.input(),
      convOriginal.filter(),
      convOriginal.bias(),
      convOriginal.dilation_h_factor(),
      convOriginal.dilation_w_factor(),
      convOriginal.fused_activation_function(),
      convOriginal.padding(),
      convOriginal.stride_h(),
      convOriginal.stride_w() );

    auto stridedSliceInput = convOriginal.input();

    // Extract args from the op
    auto inputType =
    stridedSliceInput.getType().dyn_cast<RankedTensorType>();
    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);
    auto inputDepth = inputType.getDimSize(3);

    int32_t offset = (inputHeight*inputWidth*inputDepth)/2;

    int32_t beginAttr [4] = {0,0,0,0};
    auto beginConstantOp =
      rewriter.create<arith::ConstantOp>(convOriginal.getLoc(), rewriter.getI32TensorAttr(beginAttr));

    int32_t endAttr [4] = {0,5,5,0};
    auto endConstantOp =
      rewriter.create<arith::ConstantOp>(convOriginal.getLoc(), rewriter.getI32TensorAttr(endAttr));
  
    int32_t stridesAttr [4] = {1,1,1,1};
    auto stridesConstantOp =
      rewriter.create<arith::ConstantOp>(convOriginal.getLoc(), rewriter.getI32TensorAttr(stridesAttr));

    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
    begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask = 0;
          
    auto stridedSliceOp0  = rewriter.create<TFL::StridedSliceOp>(
      convOriginal.getLoc(), convOriginal.getType(),
          convReplacement, beginConstantOp,endConstantOp,stridesConstantOp,  begin_mask,  end_mask, 
     ellipsis_mask,  new_axis_mask,  shrink_axis_mask);

    SmallVector<Value> stridedSliceOps;
    stridedSliceOps.push_back(stridedSliceOp0.getResult());

    auto stridedSliceOp1  = rewriter.create<TFL::StridedSliceOp>(
      convOriginal.getLoc(), convOriginal.getType(),
          convReplacement, beginConstantOp,endConstantOp,stridesConstantOp,  begin_mask,  end_mask, 
     ellipsis_mask,  new_axis_mask,  shrink_axis_mask);

    stridedSliceOps.push_back(stridedSliceOp1.getResult());

    RankedTensorType newOutputType = RankedTensorType::get(
        convOriginal.output().getType().cast<RankedTensorType>().getShape(),
        convOriginal.output().getType().cast<ShapedType>().getElementType());

    StringRef fused_activation_function = "NONE";

    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
          convOriginal.getLoc(), newOutputType, stridedSliceOps, 0, fused_activation_function);

    rewriter.replaceOp(convOriginal, newConcatOp.output());

    return success();
  }
};

void OpSplit::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();
  patterns.insert<OpSplitPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace xcore
} // namespace mlir
