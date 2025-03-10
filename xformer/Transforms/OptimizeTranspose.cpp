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

static SmallVector<int64_t, 4>
computeInversePermutation(const SmallVector<int64_t, 4> &permVec) {
  SmallVector<int64_t, 4> invPermVec(permVec.size());
  for (size_t i = 0; i < permVec.size(); ++i) {
    invPermVec[permVec[i]] = i;
  }
  return invPermVec;
}

struct FoldTrReFCPattern : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {
    // Match the pattern: transpose -> reshape -> fully_connected

    // Check that the input to fully_connected is a reshape
    auto reshapeOp = fcOp.getInput().getDefiningOp<TFL::ReshapeOp>();
    if (!reshapeOp || !reshapeOp->hasOneUse())
      return failure();

    // Check that the input to reshape is a transpose
    auto transposeOp = reshapeOp.getInput().getDefiningOp<TFL::TransposeOp>();
    if (!transposeOp || !transposeOp->hasOneUse())
      return failure();

    // Ensure that the fully_connected op has a single use
    if (!fcOp->hasOneUse())
      return failure();

    // Get permutation attribute from transpose
    DenseIntElementsAttr permAttr;
    if (!matchPattern(transposeOp.getPerm(), m_Constant(&permAttr)))
      return failure();

    SmallVector<int64_t, 4> permVec = utils::denseToVector<int64_t>(permAttr);

    // Compute inverse permutation vector
    SmallVector<int64_t, 4> invPermVec = computeInversePermutation(permVec);

    // Get input shape
    auto reshapeInputType =
        reshapeOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (!reshapeInputType)
      return failure();

    // Transpose the filter weights
    Value filter = fcOp.getFilter();
    Value bias = fcOp.getBias();

    {
      // Get filter type and shape
      auto filterType = filter.getType().dyn_cast<RankedTensorType>();
      if (!filterType)
        return failure();

      auto filterShape = filterType.getShape();
      SmallVector<int64_t, 4> filterShapeVec(filterShape.begin(),
                                             filterShape.end());

      int64_t outputSize = filterShapeVec[0];

      // Reshape the filter to reshapeInputShape[1:] + [outputSize]
      auto reshapeInputShape = reshapeInputType.getShape();
      SmallVector<int64_t, 4> reshapedFilterShapeVec(
          reshapeInputShape.begin() + 1, reshapeInputShape.end());
      reshapedFilterShapeVec.push_back(outputSize);

      // Prepare permutation vector for the filter (excluding batch dimension)
      SmallVector<int64_t, 4> filterPermVec;
      for (size_t i = 1; i < invPermVec.size(); ++i) {
        filterPermVec.push_back(invPermVec[i] - 1);
      }
      filterPermVec.push_back(invPermVec.size() - 1);

      Value finalFilter;
      if (failed(utils::reshapeTransposeReshape(
              rewriter, filter, reshapedFilterShapeVec, filterPermVec,
              filterShapeVec, finalFilter))) {
        llvm::outs() << "Failed to reshape filter\n";
        return failure();
      }

      filter = finalFilter;
    }

    // Bias remains the same; no need to adjust it

    // Create new reshape op (from transpose's input to reshape's output)
    Value newInput = transposeOp.getInput();

    auto newReshapeOp = rewriter.create<TFL::ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getResult().getType(), newInput,
        reshapeOp.getShape());

    // Create new fully_connected op with adjusted filter
    auto newFullyConnectedOp = rewriter.create<TFL::FullyConnectedOp>(
        fcOp.getLoc(), fcOp.getResult(0).getType(), newReshapeOp.getResult(),
        filter, bias, fcOp.getFusedActivationFunctionAttr(),
        fcOp.getWeightsFormatAttr(), fcOp.getKeepNumDimsAttr(),
        fcOp.getAsymmetricQuantizeInputsAttr());

    // Replace the original fully_connected op with the new one
    rewriter.replaceOp(fcOp, newFullyConnectedOp.getResults());

    // Erase the old reshape and transpose ops if they are no longer used
    if (reshapeOp.use_empty())
      rewriter.eraseOp(reshapeOp);
    if (transposeOp.use_empty())
      rewriter.eraseOp(transposeOp);

    return success();
  }
};

struct FoldDoubleTransposePattern : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the TransposeOp has a single use
    if (!transposeOp->hasOneUse())
      return failure();
    // Check if the user operation is a transpose op
    Operation *userOp = *transposeOp->getUsers().begin();
    auto userTransposeOp = dyn_cast<TFL::TransposeOp>(userOp);
    if (!userTransposeOp)
      return failure();
    // Get the permutation used in the transposes
    DenseIntElementsAttr perm0;
    DenseIntElementsAttr perm1;
    if (!matchPattern(transposeOp.getPerm(), m_Constant(&perm0)) ||
        !matchPattern(userTransposeOp.getPerm(), m_Constant(&perm1)))
      return failure();

    SmallVector<int32_t, 4> permVec;
    for (auto val : perm1.getValues<int32_t>()) {
      permVec.push_back(perm0.getValues<int32_t>()[val]);
    }
    // Create perm constant op
    auto permType = RankedTensorType::get(
        {static_cast<int64_t>(permVec.size())}, rewriter.getIntegerType(32));

    auto permAttr = DenseIntElementsAttr::get(permType, permVec);
    auto permConstOp =
        rewriter.create<TFL::ConstOp>(transposeOp.getLoc(), permType, permAttr);

    // Create new transposeOp
    auto newTransposeOp = rewriter.create<TFL::TransposeOp>(
        transposeOp.getLoc(), userTransposeOp.getType(), transposeOp.getInput(),
        permConstOp.getResult());

    rewriter.replaceOp(userOp, newTransposeOp.getResult());
    rewriter.eraseOp(transposeOp);
    return success();
  }
};

// Replace TransposeOp with ReshapeOp if equivalent
// Transpose is equivalent to reshape if we only permute consecutive dimensions
// and only one of those permuted dimensions isn't of size 1
struct FoldTransposeToReshapePattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr perm;
    if (!matchPattern(transposeOp.getPerm(), m_Constant(&perm)))
      return failure();

    SmallVector<int32_t, 4> permVec;
    for (auto val : perm.getValues<int32_t>()) {
      permVec.push_back(val);
    }
    // get input shape
    auto inputType =
        transposeOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    for (size_t i = 0; i < permVec.size(); ++i) {
      if (permVec[i] == i)
        continue;
      // check if all shapes between i and permVec[i] are 1, except for at most
      // one
      int s = permVec[i] < i ? permVec[i] : i;
      int e = permVec[i] < i ? i : permVec[i];
      bool foundNonOne = false;
      for (size_t j = s; j <= e; ++j) {
        if (inputShape[j] != 1) {
          if (foundNonOne)
            return failure();
          foundNonOne = true;
        }
      }
    }
    // get output shape
    auto outputType =
        transposeOp.getResult().getType().dyn_cast<RankedTensorType>();
    // convert to small vector
    SmallVector<int64_t, 4> outputShape;
    for (auto val : outputType.getShape()) {
      outputShape.push_back(val);
    }
    // Create new shape constant op
    auto newShapeConstOp =
        utils::createShapeConstOp(rewriter, transposeOp.getLoc(), outputShape);
    // assume transpose can be replaced with reshape
    auto reshapeOp = rewriter.create<TFL::ReshapeOp>(
        transposeOp.getLoc(), transposeOp.getType(), transposeOp.getInput(),
        newShapeConstOp);

    rewriter.replaceOp(transposeOp, reshapeOp.getResult());
    return success();
  }
};

struct FoldFCReTrPattern : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Match the pattern: fully_connected -> reshape -> transpose
    auto reshapeOp = transposeOp.getInput().getDefiningOp<TFL::ReshapeOp>();
    if (!reshapeOp || !reshapeOp->hasOneUse())
      return failure();

    auto fullyConnectedOp =
        reshapeOp.getInput().getDefiningOp<TFL::FullyConnectedOp>();
    if (!fullyConnectedOp || !fullyConnectedOp->hasOneUse())
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
    auto transposeOutputShape = transposeOutputType.getShape();

    if (fcOutputShape.empty() || reshapeOutputShape.empty())
      return failure(); // Expecting non-scalar tensors

    if (fcOutputShape[0] != reshapeOutputShape[0])
      return failure(); // Batch dimension changed in reshape

    // Get permutation attribute from transpose
    DenseIntElementsAttr permAttr;
    if (!matchPattern(transposeOp.getPerm(), m_Constant(&permAttr)))
      return failure();

    SmallVector<int64_t, 4> permVec = utils::denseToVector<int64_t>(permAttr);

    // Compute inverse permutation vector
    SmallVector<int64_t, 4> invPermVec = computeInversePermutation(permVec);

    // Check if batch dimension remains at position 0 after transpose
    if (invPermVec.empty() || invPermVec[0] != 0)
      return failure();

    // Prepare to transform the filter and bias
    Value filter = fullyConnectedOp.getFilter();
    Value bias = fullyConnectedOp.getBias();

    // Process bias
    {
      // Get bias type and shape
      auto biasType = bias.getType().dyn_cast<RankedTensorType>();
      if (!biasType)
        return failure();
      auto biasShape = biasType.getShape();
      SmallVector<int64_t, 4> biasShapeVec(biasShape.begin(), biasShape.end());

      // Use transpose output shape as reshape shape for bias
      SmallVector<int64_t, 4> transposeOutputShapeVec(
          transposeOutputShape.begin(), transposeOutputShape.end());

      Value finalBias;
      if (failed(utils::reshapeTransposeReshape(
              rewriter, bias, transposeOutputShapeVec, invPermVec, biasShapeVec,
              finalBias)))
        return failure();

      bias = finalBias;
    }

    // Process filter
    {
      // Get filter type and shape
      auto filterType = filter.getType().dyn_cast<RankedTensorType>();
      if (!filterType)
        return failure();

      auto filterShape = filterType.getShape();
      SmallVector<int64_t, 4> filterShapeVec(filterShape.begin(),
                                             filterShape.end());

      // Compute new filter reshape shape
      SmallVector<int64_t, 4> filterOutShapeVec = {filterShape[1]};
      filterOutShapeVec.insert(filterOutShapeVec.end(),
                               transposeOutputShape.begin() + 1,
                               transposeOutputShape.end());

      Value finalFilter;
      if (failed(utils::reshapeTransposeReshape(rewriter, filter,
                                                filterOutShapeVec, invPermVec,
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

    // Create new reshape op with the output type of the original transpose op
    SmallVector<int64_t, 4> originalShapeVec(transposeOutputShape.begin(),
                                             transposeOutputShape.end());
    Value newShapeConstOp = utils::createShapeConstOp(
        rewriter, transposeOp.getLoc(), originalShapeVec);

    // Create new reshape op
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
    auto permConstOp =
        rewriter.create<TFL::ConstOp>(concatOp.getLoc(), permType, permAttr);

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
    auto paddingOp = rewriter.create<TFL::ConstOp>(
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
  mergePatterns
      .insert<MoveTransposeForwardOverUnaryOpPattern,
              MoveTransposeForwardOverConcatOpPattern,
              FoldDoubleTransposePattern, FoldTransposeToReshapePattern>(ctx);
  if (mergeTransposeOption) {
    (void)applyPatternsAndFoldGreedily(func, std::move(mergePatterns));
  }

  // Other transpose optimizations
  RewritePatternSet patterns(ctx);

  patterns.insert<HoistTransposeWCHAbovePadPattern>(ctx);
  patterns.insert<FoldDoubleTransposePattern>(ctx);
  patterns.insert<FoldTransposeToReshapePattern>(ctx);
  // TODO - enable after transpose permutation fix
  // patterns.insert<FoldFCReTrPattern>(ctx);
  // patterns.insert<FoldTrReFCPattern>(ctx);
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
