// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Options.h"

#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir::xcore {

namespace {
// Optimize TFL Transpose.
struct OptimizeTranspose
    : public PassWrapper<OptimizeTranspose, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTranspose)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-optimize-transpose"; }
  StringRef getDescription() const final { return "Optimize TFL Transpose."; }
  void runOnOperation() override;
};

struct FoldTransposeIntoFullyConnectedPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Match the pattern: fully_connected -> reshape -> transpose
    auto reshapeOp = transposeOp.getInput().getDefiningOp<TFL::ReshapeOp>();
    if (!reshapeOp || !reshapeOp->getResult(0).hasOneUse())
      return failure();

    auto fullyConnectedOp =
        reshapeOp.getInput().getDefiningOp<TFL::FullyConnectedOp>();
    if (!fullyConnectedOp || !fullyConnectedOp->getResult(0).hasOneUse())
      return failure();

    // Get types and shapes
    auto fcInputType =
        fullyConnectedOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto fcOutputType =
        fullyConnectedOp.getResult(0).getType().dyn_cast<RankedTensorType>();
    auto reshapeOutputType =
        reshapeOp.getResult().getType().dyn_cast<RankedTensorType>();
    auto transposeOutputType =
        transposeOp.getResult().getType().dyn_cast<RankedTensorType>();

    if (!fcInputType || !fcOutputType || !reshapeOutputType ||
        !transposeOutputType)
      return failure();

    // Check if the batch dimension (assumed to be dimension 0) remains
    // unchanged
    auto fcOutputShape = fcOutputType.getShape();
    auto reshapeOutputShape = reshapeOutputType.getShape();
    SmallVector<int64_t, 4> reshapeOutputShapeVec(reshapeOutputShape.begin(),
                                                  reshapeOutputShape.end());

    if (reshapeOutputShape[0] != 1) {
      return failure();
    }

    if (fcOutputShape.empty() || reshapeOutputShape.empty())
      return failure(); // Expecting non-scalar tensors

    if (fcOutputShape[0] != reshapeOutputShape[0])
      return failure(); // Batch dimension changed in reshape

    // Check if transpose does not affect the batch dimension
    DenseIntElementsAttr permAttr;
    if (!matchPattern(transposeOp.getPerm(), m_Constant(&permAttr)))
      return failure();

    SmallVector<int64_t, 4> permVec;
    for (auto val : permAttr.getValues<int32_t>()) {
      permVec.push_back(static_cast<int64_t>(val));
    }

    // Check if batch dimension remains at position 0 after transpose
    if (permVec.empty() || permVec[0] != 0)
      return failure();

    // Prepare to transform the filter and bias
    Value filter = fullyConnectedOp.getFilter();
    Value bias = fullyConnectedOp.getBias();

    // Process bias
    {
      // Ensure bias is produced by a TFL::QConstOp
      auto biasQConstOp = bias.getDefiningOp<TFL::QConstOp>();
      if (!biasQConstOp)
        return failure();

      // Get bias type and shape
      auto biasType = bias.getType().dyn_cast<RankedTensorType>();
      if (!biasType)
        return failure();
      auto biasShape = biasType.getShape();

      SmallVector<int64_t, 4> biasShapeVec(biasShape.begin(), biasShape.end());
      Value finalBias;
      if (failed(utils::reshapeTransposeReshape(rewriter, bias,
                                                reshapeOutputShapeVec, permVec,
                                                biasShapeVec, finalBias)))
        return failure();

      // Update bias
      bias = finalBias;
    }

    // Process filter
    {
      // Ensure filter is produced by a TFL::QConstOp
      auto filterQConstOp = filter.getDefiningOp<TFL::QConstOp>();
      if (!filterQConstOp)
        return failure();

      // Get filter type and shape
      auto filterType = filter.getType().dyn_cast<RankedTensorType>();
      if (!filterType)
        return failure();
      auto filterShape = filterType.getShape();
      SmallVector<int64_t, 4> filterShapeVec(filterShape.begin(),
                                             filterShape.end());

      // same as the shape of the reshape, except for first dimension which
      // should be the first dimension of the filterShape
      SmallVector<int64_t, 4> filterOutShapeVec = {filterShape[1]};
      filterOutShapeVec.insert(filterOutShapeVec.end(),
                               reshapeOutputShapeVec.begin() + 1,
                               reshapeOutputShapeVec.end());

      Value finalFilter;
      if (failed(utils::reshapeTransposeReshape(rewriter, filter,
                                                filterOutShapeVec, permVec,
                                                filterShapeVec, finalFilter)))
        return failure();
      filter = finalFilter;
    }

    // Create new fully connected op with adjusted filter and bias
    auto newFullyConnectedOp = rewriter.create<TFL::FullyConnectedOp>(
        fullyConnectedOp.getLoc(), fcOutputType, fullyConnectedOp.getInput(),
        filter, bias, fullyConnectedOp.getFusedActivationFunctionAttr(),
        fullyConnectedOp.getWeightsFormatAttr(),
        fullyConnectedOp.getKeepNumDimsAttr(),
        fullyConnectedOp.getAsymmetricQuantizeInputsAttr());

    // create new shape from the shape of the original transpose op
    auto originalShape =
        transposeOp.getResult().getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t, 4> originalShapeVec(originalShape.begin(),
                                             originalShape.end());
    Value newShapeConstOp = utils::createShapeConstOp(
        rewriter, transposeOp.getLoc(), originalShapeVec);

    // Create new reshape op with the output type of the original transpose op
    auto newReshapeOp = rewriter.create<TFL::ReshapeOp>(
        reshapeOp.getLoc(), transposeOutputType,
        newFullyConnectedOp.getResult(0), newShapeConstOp);

    // Replace the original transpose op with the new reshape op
    rewriter.replaceOp(transposeOp, newReshapeOp.getResult());

    return success();
  }
};

struct MoveTransposeForwardOverUnaryOpPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the TransposeOp has a single use
    if (!transposeOp->hasOneUse())
      return failure();

    Operation *userOp = *transposeOp->getUsers().begin();

    // Check if the user operation is a unary op that can commute with
    // transpose
    if (!isa<TFL::AbsOp, TFL::NegOp, TFL::ReluOp, TFL::Relu6Op, TFL::QuantizeOp,
             TFL::LeakyReluOp, TFL::TanhOp, TFL::LogisticOp>(userOp))
      return failure();

    // Get the types of the input and output tensors
    auto transposeInputType =
        transposeOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto transposeOutputType =
        transposeOp.getType().dyn_cast<RankedTensorType>();
    if (!transposeInputType || !transposeOutputType)
      return failure();

    // Get the permutation used in the transpose
    Value perm = transposeOp.getPerm();

    Value newUnaryOpResult;
    auto loc = userOp->getLoc();
    auto input = transposeOp.getInput();

    // Retrieve the original unary operation's output type
    auto originalUnaryOutputType =
        userOp->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!originalUnaryOutputType)
      return failure();

    // Create a new output type for the unary op with the same shape as
    // 'input' and the same element type as the original output type
    auto newUnaryOutputType = RankedTensorType::get(
        transposeInputType.getShape(), originalUnaryOutputType.getElementType(),
        originalUnaryOutputType.getEncoding());

    if (auto quantizeOp = dyn_cast<TFL::QuantizeOp>(userOp)) {
      // For QuantizeOp, create new QuantizeOp with input as 'input' and
      // output type adjusted

      // Create new QuantizeOp with adjusted output type
      newUnaryOpResult = rewriter.create<TFL::QuantizeOp>(
          loc, newUnaryOutputType, input, quantizeOp.getQtypeAttr());

    } else if (auto absOp = dyn_cast<TFL::AbsOp>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::AbsOp>(loc, newUnaryOutputType, input);
    } else if (auto negOp = dyn_cast<TFL::NegOp>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::NegOp>(loc, newUnaryOutputType, input);
    } else if (auto reluOp = dyn_cast<TFL::ReluOp>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::ReluOp>(loc, newUnaryOutputType, input);
    } else if (auto relu6Op = dyn_cast<TFL::Relu6Op>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::Relu6Op>(loc, newUnaryOutputType, input);
    } else if (auto leakyReluOp = dyn_cast<TFL::LeakyReluOp>(userOp)) {
      newUnaryOpResult = rewriter.create<TFL::LeakyReluOp>(
          loc, newUnaryOutputType, input, leakyReluOp.getAlphaAttr());
    } else if (auto tanhOp = dyn_cast<TFL::TanhOp>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::TanhOp>(loc, newUnaryOutputType, input);
    } else if (auto logisticOp = dyn_cast<TFL::LogisticOp>(userOp)) {
      newUnaryOpResult =
          rewriter.create<TFL::LogisticOp>(loc, newUnaryOutputType, input);
    } else {
      // This should not happen as we checked the op type earlier
      return failure();
    }

    // Create a new Transpose operation after the unary operation
    auto newTransposeOp = rewriter.create<TFL::TransposeOp>(
        transposeOp.getLoc(), originalUnaryOutputType, newUnaryOpResult, perm);

    // Replace the original user operation's result with the new transpose
    // result
    rewriter.replaceOp(userOp, newTransposeOp.getResult());

    // Remove the original TransposeOp
    rewriter.eraseOp(transposeOp);

    return success();
  }
};

struct FoldCancellableTransposePattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Defining op must be transpose
    auto transposeOp =
        dyn_cast_or_null<TFL::TransposeOp>(op.getInput().getDefiningOp());
    if (!transposeOp) {
      return failure();
    }

    // Get transpose permutations
    DenseIntElementsAttr perm0;
    DenseIntElementsAttr perm1;
    if (!matchPattern(op.getPerm(), m_Constant(&perm0)) ||
        !matchPattern(transposeOp.getPerm(), m_Constant(&perm1))) {
      return failure();
    }

    // Do permutation indices cancel each other?
    if (!TF::AreCancellablePermutations(perm0, perm1)) {
      return failure();
    }

    rewriter.replaceOp(op, transposeOp.getInput());

    return success();
  }
};
struct MoveTransposeForwardOverConcatOpPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {
    // Get all input operands
    auto inputs = concatOp.getValues();

    // Check that all inputs are TransposeOps with the same permutation
    SmallVector<Value, 4> newInputs;
    DenseIntElementsAttr commonPermAttr;
    for (auto input : inputs) {
      auto transposeOp = input.getDefiningOp<TFL::TransposeOp>();
      if (!transposeOp)
        return failure();

      // Get permutation attribute
      DenseIntElementsAttr permAttr;
      if (!matchPattern(transposeOp.getPerm(), m_Constant(&permAttr)))
        return failure();

      // Check if the permutation is the same as others
      if (commonPermAttr) {
        if (permAttr != commonPermAttr)
          return failure();
      } else {
        commonPermAttr = permAttr;
      }

      // Collect the inputs to the transpose ops
      newInputs.push_back(transposeOp.getInput());
    }

    // Get the permutation vector
    SmallVector<int32_t, 4> permVec;
    for (auto val : commonPermAttr.getValues<int32_t>()) {
      permVec.push_back(val);
    }

    // Compute the inverse permutation
    SmallVector<int32_t, 4> invPerm(permVec.size());
    for (size_t i = 0; i < permVec.size(); ++i) {
      invPerm[permVec[i]] = i;
    }

    // Adjust the axis according to the inverse permutation
    int32_t oldAxis = concatOp.getAxis();
    int64_t rank = permVec.size();
    if (oldAxis < 0) {
      oldAxis += rank;
    }
    if (oldAxis < 0 || oldAxis >= rank) {
      return failure(); // Invalid axis
    }
    int32_t newAxis = permVec[oldAxis];

    // Collect input types and compute the new result type
    SmallVector<RankedTensorType, 4> inputTypes;
    for (auto input : newInputs) {
      auto inputType = input.getType().dyn_cast<RankedTensorType>();
      if (!inputType) {
        return failure();
      }
      inputTypes.push_back(inputType);
    }

    // Ensure all input types have the same rank
    for (auto type : inputTypes) {
      if (type.getRank() != rank) {
        return failure();
      }
    }

    // Compute the shape of the concatenated tensor
    SmallVector<int64_t, 4> resultShape(inputTypes[0].getShape().begin(),
                                        inputTypes[0].getShape().end());
    for (size_t i = 1; i < inputTypes.size(); ++i) {
      auto shape = inputTypes[i].getShape();
      resultShape[newAxis] += shape[newAxis];
    }

    // Create the new ConcatenationOp with the correct result type and axis
    auto elementType = inputTypes[0].getElementType();
    auto newConcatType = RankedTensorType::get(resultShape, elementType);
    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        concatOp.getLoc(), newConcatType, newInputs,
        rewriter.getI32IntegerAttr(newAxis),
        concatOp.getFusedActivationFunctionAttr());

    // Create the permutation constant with correct data types
    auto permType = RankedTensorType::get(
        {static_cast<int64_t>(permVec.size())}, rewriter.getIntegerType(32));
    auto permAttr = DenseIntElementsAttr::get(permType, permVec);
    auto permConstOp = rewriter.create<arith::ConstantOp>(concatOp.getLoc(),
                                                          permType, permAttr);

    // Create the new TransposeOp with the original output type
    auto newTransposeOp = rewriter.create<TFL::TransposeOp>(
        concatOp.getLoc(), concatOp.getType(), newConcatOp.getResult(),
        permConstOp.getResult());

    rewriter.replaceOp(concatOp, newTransposeOp.getResult());
    return success();
  }
};

struct HoistTransposeWCHAbovePadPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Defining op must be pad
    auto padOp = dyn_cast_or_null<TFL::PadOp>(op.getInput().getDefiningOp());
    if (!padOp) {
      return failure();
    }

    // Get transpose permutation
    DenseIntElementsAttr perm;
    if (!matchPattern(op.getPerm(), m_Constant(&perm))) {
      return failure();
    }

    // Confirm transpose permutation is 0,2,3,1 i.e., NWCH
    // Remnants of Pytorch to TFlite conversion
    auto permVal = perm.getValues<int32_t>();
    if (perm.size() != 4 || permVal[0] != 0 || permVal[1] != 2 ||
        permVal[2] != 3 || permVal[3] != 1) {
      return failure();
    }

    // Get padding val
    DenseIntElementsAttr pad;
    if (!matchPattern(padOp.getPadding(), m_Constant(&pad))) {
      return failure();
    }

    // Confirm padding is only in last two dimensions
    auto padVal = pad.getValues<int32_t>();
    if (padVal[{0, 0}] != 0 || padVal[{0, 1}] != 0 || padVal[{1, 0}] != 0 ||
        padVal[{1, 1}] != 0 || padVal[{2, 0}] != 1 || padVal[{2, 1}] != 1 ||
        padVal[{3, 0}] != 1 || padVal[{3, 1}] != 1) {
      return failure();
    }

    // Create new TransposeOp
    auto padInputShape =
        padOp.getInput().getType().cast<RankedTensorType>().getShape();
    auto tranposeResultType = RankedTensorType::get(
        {padInputShape[0], padInputShape[2], padInputShape[3],
         padInputShape[1]},
        padOp.getInput().getType().cast<ShapedType>().getElementType());
    auto newTranspose = rewriter.create<TFL::TransposeOp>(
        padOp.getLoc(), tranposeResultType, padOp.getInput(), op.getPerm());

    // Create new padding attr with spatial dimensions
    std::vector<int32_t> paddingValues{0, 0, 1, 1, 1, 1, 0, 0};
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4, 2}, rewriter.getI32Type()), paddingValues);
    auto paddingOp = rewriter.create<arith::ConstantOp>(
        padOp->getLoc(), RankedTensorType::get({4, 2}, rewriter.getI32Type()),
        paddingAttr);
    auto newPad = rewriter.create<TFL::PadOp>(
        padOp.getLoc(), op.getOutput().getType(), newTranspose, paddingOp);

    rewriter.replaceOp(op, newPad.getOutput());
    return success();
  }
};

struct FoldTransposeWCHToInput : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Get transpose permutation
    DenseIntElementsAttr perm;
    if (!matchPattern(op.getPerm(), m_Constant(&perm))) {
      return failure();
    }

    // Confirm transpose permutation is 0,2,3,1 i.e., NWCH
    // Remnants of Pytorch to TFlite conversion
    auto permVal = perm.getValues<int32_t>();
    if (perm.size() != 4 || permVal[0] != 0 || permVal[1] != 2 ||
        permVal[2] != 3 || permVal[3] != 1) {
      return failure();
    }

    // If input to the transpose is block arg, and block arg has only one use,
    // we can fold the transpose
    if (auto blockArg = op.getInput().dyn_cast<BlockArgument>()) {
      if (blockArg.hasOneUse()) {
        auto funcOp = cast<func::FuncOp>(blockArg.getOwner()->getParentOp());

        // Set function type to the transpose output type as we are changing
        // the input
        FunctionType funcType = funcOp.getFunctionType();
        llvm::SmallVector<Type, 4> newInputTypes(funcType.getInputs().begin(),
                                                 funcType.getInputs().end());
        newInputTypes[blockArg.getArgNumber()] = op.getOutput().getType();
        auto newFuncType = FunctionType::get(
            rewriter.getContext(), newInputTypes, funcOp.getResultTypes());
        funcOp.setType(newFuncType);

        // Set block arg type to the transpose output type
        blockArg.setType(op.getOutput().getType());

        // Remove transpose
        rewriter.replaceOp(op, op.getInput());
      }
    }

    return success();
  }
};

void OptimizeTranspose::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  // Try to merge transpose -> ops -> inverse transpose
  RewritePatternSet mergePatterns(ctx);
  mergePatterns.insert<MoveTransposeForwardOverUnaryOpPattern,
                       MoveTransposeForwardOverConcatOpPattern,
                       FoldCancellableTransposePattern>(ctx);
  if (mergeTransposeOption) {
    (void)applyPatternsAndFoldGreedily(func, std::move(mergePatterns));
  }

  // Other transpose optimizations
  RewritePatternSet patterns(ctx);

  patterns.insert<HoistTransposeWCHAbovePadPattern>(ctx);
  patterns.insert<FoldCancellableTransposePattern>(ctx);
  patterns.insert<FoldTransposeIntoFullyConnectedPattern>(ctx);
  if (allowInputModificationOption) {
    patterns.insert<FoldTransposeWCHToInput>(ctx);
  }

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OptimizeTranspose pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeTransposePass() {
  return std::make_unique<OptimizeTranspose>();
}

static PassRegistration<OptimizeTranspose> pass;

} // namespace mlir::xcore
