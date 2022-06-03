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
// Replace  Concat
struct ReplaceConcat
    : public PassWrapper<ReplaceConcat, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct ReplaceConcatPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {

    auto input0 = concatOp.values().operator[](0);
    auto input1 = concatOp.values().operator[](1);

    auto inputType0 = input0.getType().dyn_cast<RankedTensorType>();
    auto inputType1 = input1.getType().dyn_cast<RankedTensorType>();
    
    auto inputHeight0 = inputType0.getDimSize(1);
    auto inputWidth0 = inputType0.getDimSize(2);
    auto inputDepth0 = inputType0.getDimSize(3);

    auto inputHeight1 = inputType1.getDimSize(1);
    auto inputWidth1 = inputType1.getDimSize(2);
    auto inputDepth1 = inputType1.getDimSize(3);

    auto image_geom0 = nn::ImageGeometry(inputHeight0, inputWidth0,
                                        static_cast<int>(inputDepth0));

    auto window_geom0 =
        nn::WindowGeometry({static_cast<int>(inputHeight0), static_cast<int>(inputWidth0), static_cast<int>(inputDepth0)},
                           {0, 0}, {1, 1, 1}, {1, 1});

    auto image_geom1 = nn::ImageGeometry(inputHeight1, inputWidth1,
                                        static_cast<int>(inputDepth1));

    auto window_geom1 =
        nn::WindowGeometry({static_cast<int>(inputHeight1), static_cast<int>(inputWidth1), static_cast<int>(inputDepth1)},
                           {0, 0}, {1, 1, 1}, {1, 1});
    
    

    nn::ImToColValid::Params imToColParams0(image_geom0, window_geom0,static_cast<int>(inputDepth0));
    std::string mfStr0 = imToColParams0.serialise<nn::ImToColValid::Params>();

    nn::ImToColValid::Params imToColParams1(image_geom1, window_geom1,static_cast<int>(inputDepth1));
    std::string mfStr1 = imToColParams1.serialise<nn::ImToColValid::Params>();

    auto binaryObjectConcatOp = rewriter.create<ConcatOp>(
        concatOp.getLoc(), concatOp.getType(),
        input0, input1,  rewriter.getStringAttr(mfStr0), rewriter.getStringAttr(mfStr1));
    rewriter.replaceOp(concatOp, binaryObjectConcatOp.output());

    return success();
  }
};

void ReplaceConcat::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat>
    pass("xcore-replace-concat",
         "Replace TFL Concat with Concat for XCore.");

} // namespace xcore
} // namespace mlir
