// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace TFL StridedSlice with StridedSlice for XCore.
struct InsertStridedSliceConcat
    : public PassWrapper<InsertStridedSliceConcat, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct InsertStridedSliceConcatPattern
    : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv2DOp,
                                PatternRewriter &rewriter) const override {

    bool op_spilt = true;

    auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
        conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(),
        conv2DOp.filter(),
        conv2DOp.bias(),
        conv2DOp.dilation_h_factor(),
        conv2DOp.dilation_w_factor(),
        conv2DOp.fused_activation_function(),
        conv2DOp.padding(),
        conv2DOp.stride_h(),
        conv2DOp.stride_w(),
        conv2DOp.
        );
    
    

    rewriter.replaceOp(conv2DOp, newConv2DOp.output());

    return success();
  }
};

void InsertStridedSliceConcat::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InsertStridedSliceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the InsertStridedSliceConcat pass.
std::unique_ptr<OperationPass<FuncOp>> createInsertStridedSliceConcatPass() {
  return std::make_unique<InsertStridedSliceConcat>();
}

static PassRegistration<InsertStridedSliceConcat>
    pass("insert-stridedslice-concat",
         "InsertStridedSliceConcat.");

} // namespace xcore
} // namespace mlir
