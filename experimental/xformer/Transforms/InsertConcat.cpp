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
// Insert  Concat
struct InsertConcat
    : public PassWrapper<InsertConcat,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-insert-concat"; }
  StringRef getDescription() const final {
    return "Insert TFL Concat.";
  }
  void runOnOperation() override;
};



struct InsertConcatPattern
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

    auto simpleSliceOp0  = rewriter.create<SimpleSliceOp>(
      convOriginal.getLoc(), convOriginal.getType(),
          convReplacement, rewriter.getI32IntegerAttr(offset)) ; 

    // SmallVector<Value> beginAttr;
    // beginAttr.push_back(0);
    // beginAttr.push_back(0);

    // SmallVector<Value> endAttr;
    // endAttr.push_back(4);
    // endAttr.push_back(4);

    // SmallVector<Value> stridesAttr;
    // endAttr.push_back(1);
    // endAttr.push_back(1);
          
    // auto simpleSliceOp0  = rewriter.create<TFL::StridedSliceOp>(
    //   convOriginal.getLoc(), convOriginal.getType(),
    //       convReplacement, beginAttr,endAttr,stridesAttr) ; 

    SmallVector<Value> stridedSliceOps;
    stridedSliceOps.push_back(simpleSliceOp0.getResult());

    auto simpleSliceOp1  = rewriter.create<SimpleSliceOp>(
      convOriginal.getLoc(), convOriginal.getType(),
          convReplacement, rewriter.getI32IntegerAttr(offset)) ; 

    stridedSliceOps.push_back(simpleSliceOp1.getResult());

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

void InsertConcat::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();
  patterns.insert<InsertConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the InsertConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createInsertConcatPass() {
  return std::make_unique<InsertConcat>();
}

static PassRegistration<InsertConcat> pass;

} // namespace xcore
} // namespace mlir
