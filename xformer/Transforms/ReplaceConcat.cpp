// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

// 13 because flexbuffers are allocated in multiples of 16 bytes, and the op
// already allocates 2 integers and a boolean on top of CONCAT_OP_MAX_INPUTS * 4
// bytes. This will use 64 bytes per concat op.
//
// This number must match kMaxNumInputs in the ConcatOp definition in the
// runtime.
namespace mlir::xcore {

namespace {

// Replace TFL Concatenate with Concat for XCore.
struct ReplaceConcat
    : public PassWrapper<ReplaceConcat, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concatenate with Concat for XCore.";
  }
  void runOnOperation() override;
};

bool isConcatConvertible(TFL::ConcatenationOp concatOp) {
  auto values = concatOp.getValues();
  for (int i = 0; i < values.size(); i++) {
    auto inputType = values[i].getType().cast<RankedTensorType>();
    if (!inputType.hasStaticShape())
      return false;
  }
  if (concatOp.getFusedActivationFunction() != "NONE")
    return false;
  return true;
}

struct SplitConcatPattern : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {

    // No need to split if the op won't be converted to XC concat anyway
    if (!isConcatConvertible(concatOp))
      return failure();
    mlir::Operation::operand_range values = concatOp.getValues();
    int num_inputs = values.size();
    if (num_inputs <= CONCAT_OP_MAX_INPUTS)
      return failure();

    auto outputType = concatOp.getOutput().getType().cast<RankedTensorType>();
    Type elementType = outputType.getElementType();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    const int axis = concatOp.getAxis();

    int axisShape = 0;
    for (int i = 0; i < CONCAT_OP_MAX_INPUTS; i++) {
      auto inputType = values[i].getType().cast<RankedTensorType>();
      axisShape += inputType.getShape()[axis];
    }
    int rank = outputType.getRank();
    int64_t newShape[rank];
    for (int i = 0; i < rank; i++)
      newShape[i] = outputShape[i];
    newShape[axis] = axisShape;

    std::vector<int64_t> newShapeVec(newShape, newShape + rank);
    auto newOutputType =
        RankedTensorType::get(ArrayRef<int64_t>(newShapeVec), elementType);
    auto firstValues = values.take_front(CONCAT_OP_MAX_INPUTS);

    auto first_concatOp = rewriter.create<TFL::ConcatenationOp>(
        concatOp.getLoc(), newOutputType, firstValues, concatOp.getAxis(),
        "NONE");

    Value first_output = first_concatOp.getResult();
    OperandRange remainingValues = values.drop_front(CONCAT_OP_MAX_INPUTS);
    SmallVector<Value, 4> remainingValuesVec;
    remainingValuesVec.push_back(first_output);
    for (auto value : remainingValues)
      remainingValuesVec.push_back(value);

    auto remaining_concatOp = rewriter.create<TFL::ConcatenationOp>(
        concatOp.getLoc(), outputType, remainingValuesVec, concatOp.getAxis(),
        concatOp.getFusedActivationFunction());

    rewriter.replaceOp(concatOp, remaining_concatOp.getResult());
    return success();
  }
};

struct ReplaceConcatPattern : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {

    if (!isConcatConvertible(concatOp))
      return failure();

    auto values = concatOp.getValues();
    int num_inputs = values.size();
    if (num_inputs > CONCAT_OP_MAX_INPUTS)
      return failure();

    ArrayRef<int64_t> inputShapes[CONCAT_OP_MAX_INPUTS];
    for (int i = 0; i < num_inputs; i++) {
      auto inputType = values[i].getType().cast<RankedTensorType>();
      inputShapes[i] = inputType.getShape();
    }

    auto outputType = concatOp.getOutput().getType().cast<RankedTensorType>();
    Type elementType = outputType.getElementType();

    int axis = concatOp.getAxis();
    const int rank = outputType.getRank();
    if (axis < 0)
      axis = rank + axis;

    const size_t dtype_size = utils::getTypeSize(elementType);
    int num_copies = 1;
    for (int i = 0; i < axis; i++) {
      num_copies *= outputType.getShape()[i];
    }

    bool isVpu = true;
    int32_t sizes[CONCAT_OP_MAX_INPUTS];
    for (int i = 0; i < num_inputs; i++) {
      sizes[i] = dtype_size;
      for (int j = axis; j < rank; j++)
        sizes[i] *= inputShapes[i][j];
      if (sizes[i] % 4 != 0)
        isVpu = false;
    }

    auto binaryObjectConcatOp = rewriter.create<ConcatOp>(
        concatOp.getLoc(), concatOp.getType(), values,
        rewriter.getI32IntegerAttr(num_copies), rewriter.getI32ArrayAttr(sizes),
        rewriter.getI32IntegerAttr(num_inputs), rewriter.getBoolAttr(isVpu));

    rewriter.replaceOp(concatOp, binaryObjectConcatOp.getOutput());

    return success();
  }
};

void ReplaceConcat::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<SplitConcatPattern>(ctx);
  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat> pass;

} // namespace mlir::xcore
