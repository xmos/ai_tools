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
    : public PassWrapper<ReplaceConcat, 
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
   StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concat.";
  }
  void runOnOperation() override;
};

struct ReplaceConcatPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {
    
    
    auto numInputs = concatOp.values().size();

    auto numConcats = numInputs/2 + numInputs % 2;

    auto input0 = concatOp.values().operator[](0);
    auto input1 = concatOp.values().operator[](1);

    auto inputType0 = input0.getType().dyn_cast<RankedTensorType>();
    auto inputType1 = input1.getType().dyn_cast<RankedTensorType>();
    
    auto input0Height = inputType0.getDimSize(1);
    auto input0Width = inputType0.getDimSize(2);
    auto input0Channels = inputType0.getDimSize(3);

    auto input1Height = inputType1.getDimSize(1);
    auto input1Width = inputType1.getDimSize(2);
    auto input1Channels = inputType1.getDimSize(3);

    int32_t offset0 = 0;
    int32_t offset1 = input0Height*input0Width*input0Channels;

    auto outputSize0 = (input0Height+input1Height) *(input0Width+input1Width) *(input0Channels+input1Channels);

    std::vector<int8_t> dummy(outputSize0, 0);

    ShapedType concatTensorType = RankedTensorType::get(
      outputSize0, rewriter.getI8Type());
    auto concatTensorAttr = DenseElementsAttr::get<int8_t>(concatTensorType, dummy);
    auto concatTensorOp =
      rewriter.create<arith::ConstantOp>(concatOp.getLoc(), concatTensorAttr);
    
    auto copyIntoOp0 = rewriter.create<CopyIntoOp>(
      concatOp.getLoc(), concatOp.getType(),
      input0, concatTensorOp, offset0 );

    auto copyIntoOp1  = rewriter.create<CopyIntoOp>(
      concatOp.getLoc(), concatOp.getType(),
      input1, concatTensorOp, offset1);
      
    auto connectorOp0  = rewriter.create<ConnectorOp>(
      concatOp.getLoc(), concatOp.getType(),
      copyIntoOp0, copyIntoOp1 );

  if (numConcats==1) {

    auto passThruOp  = rewriter.create<PassThruOp>(
          concatOp.getLoc(), concatOp.getType(),
            concatTensorOp, connectorOp0);
    
    rewriter.replaceOp(concatOp, passThruOp.output());

  } else if (numConcats==2){

      auto input2 = concatOp.values().operator[](2);
      auto input3 = concatOp.values().operator[](3);

      auto inputType2 = input2.getType().dyn_cast<RankedTensorType>();
      auto inputType3 = input3.getType().dyn_cast<RankedTensorType>();
      
      auto input2Height = inputType2.getDimSize(1);
      auto input2Width = inputType2.getDimSize(2);
      auto input2Channels = inputType2.getDimSize(3);

      auto input3Height = inputType3.getDimSize(1);
      auto input3Width = inputType3.getDimSize(2);
      auto input3Channels = inputType3.getDimSize(3);

      int32_t offset2 = offset1+input1Height*input1Width*input1Channels;
      int32_t offset3 = offset2+input2Height*input2Width*input2Channels;
    
      auto outputSize1 = (input2Height+input3Height) *(input2Width+input3Width) *(input2Channels+input3Channels);

      std::vector<int8_t> dummy(outputSize1, 0);

      ShapedType concatTensorType = RankedTensorType::get(
        outputSize1, rewriter.getI8Type());
      auto concatTensorAttr = DenseElementsAttr::get<int8_t>(concatTensorType, dummy);
      auto concatTensorOp =
        rewriter.create<arith::ConstantOp>(concatOp.getLoc(), concatTensorAttr);

      auto copyIntoOp2 = rewriter.create<CopyIntoOp>(
          concatOp.getLoc(), concatOp.getType(),
          input2, concatTensorOp, offset2 );

      auto copyIntoOp3  = rewriter.create<CopyIntoOp>(
          concatOp.getLoc(), concatOp.getType(),
            input3, concatTensorOp, offset3);
      
      auto connectorOp1  = rewriter.create<ConnectorOp>(
          concatOp.getLoc(), concatOp.getType(),
            copyIntoOp2, copyIntoOp3 );

      auto connectorOp2  = rewriter.create<ConnectorOp>(
          concatOp.getLoc(), concatOp.getType(),
            connectorOp0, connectorOp1 );

      auto passThruOp  = rewriter.create<PassThruOp>(
          concatOp.getLoc(), concatOp.getType(),
            concatTensorOp, connectorOp2);
    
      rewriter.replaceOp(concatOp, passThruOp.output());

    }

    return success();
  }
};

void ReplaceConcat::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat> pass;

} // namespace xcore
} // namespace mlir
