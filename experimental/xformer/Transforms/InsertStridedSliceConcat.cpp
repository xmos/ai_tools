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
                                    
    auto 

    auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
        conv2DOp.getloc(), conv2DOp.gettype(),conv2DOp.input()
        );
    rewriter.replaceop(conv2DOp,newConv2DOp.output());

    return success();
  } 
};

void InsertStridedSliceConcat::runonfunction() {
  auto *ctx = &getcontext();
  auto func = getfunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InsertStridedSliceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// creates an instance of the insertstridedsliceconcat pass.
std::unique_ptr<operationpass<funcop>> createinsertstridedsliceconcatpass() {
  return std::make_unique<insertstridedsliceconcat>();
}

static passregistration<insertstridedsliceconcat>
    pass("xcore-insert-stridedslice-concat",
         "insert tfl stridedslice and tfl concat.");

} // namespace xcore
} // namespace mlir
