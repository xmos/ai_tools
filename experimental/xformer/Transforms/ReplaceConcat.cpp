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
   StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concat.";
  }
  void runOnFunction() override;
};

struct ReplaceConcatPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {
    

    auto numInputs = concatOp.values().size();

    auto input0 = concatOp.values().operator[](0);
    auto input1 = concatOp.values().operator[](1);

    auto inputType0 = input0.getType().dyn_cast<RankedTensorType>();
    auto inputType1 = input1.getType().dyn_cast<RankedTensorType>();
    
    auto inputWidth1 = inputType0.getDimSize(1);
    auto inputWidth2 = inputType0.getDimSize(2);
    auto inputWidth3 = inputType0.getDimSize(3);
   
    auto outputType =
        concatOp.output().getType().dyn_cast<RankedTensorType>();

    // Create the tensor
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto outputDepth = outputType.getDimSize(3);

    auto outputSize = outputHeight * outputWidth * outputDepth;
    std::vector<int8_t> dummy(outputSize, 0);

    ShapedType concatTensorType = RankedTensorType::get(
        outputSize, rewriter.getI8Type());
    auto concatTensorAttr = DenseElementsAttr::get<int8_t>(concatTensorType, dummy);
    auto concatTensorOp =
      rewriter.create<ConstantOp>(concatOp.getLoc(), concatTensorAttr);
    
    int32_t offset0 = 0;
    int32_t offset1 = inputWidth1*inputWidth2*inputWidth3;

    auto copyIntoOp0 = rewriter.create<CopyIntoOp>(
        concatOp.getLoc(), concatOp.getType(),
        input0, concatTensorOp, offset0 );

    auto copyIntoOp1  = rewriter.create<CopyIntoOp>(
        concatOp.getLoc(), concatOp.getType(),
          input1, concatTensorOp, offset1);
    
    auto connectorOp0  = rewriter.create<ConnectorOp>(
        concatOp.getLoc(), concatOp.getType(),
          copyIntoOp0, copyIntoOp1 );

    auto passThruOp  = rewriter.create<PassThruOp>(
        concatOp.getLoc(), concatOp.getType(),
          concatTensorOp, connectorOp0);

    rewriter.replaceOp(concatOp, passThruOp.output());

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

static PassRegistration<ReplaceConcat> pass;

} // namespace xcore
} // namespace mlir
