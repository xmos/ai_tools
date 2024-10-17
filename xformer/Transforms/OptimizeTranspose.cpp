// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Options.h"

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

struct MoveTransposeForwardOverUnaryOpPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the TransposeOp has a single use
    if (!transposeOp->hasOneUse())
      return failure();

    Operation *userOp = *transposeOp->getUsers().begin();

    // Check if the user operation is a unary op that can commute with transpose
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

    // Create a new output type for the unary op with the same shape as 'input'
    // and the same element type as the original output type
    auto newUnaryOutputType = RankedTensorType::get(
        transposeInputType.getShape(), originalUnaryOutputType.getElementType(),
        originalUnaryOutputType.getEncoding());

    if (auto quantizeOp = dyn_cast<TFL::QuantizeOp>(userOp)) {
      // For QuantizeOp, create new QuantizeOp with input as 'input' and output
      // type adjusted

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
        transposeOp.getLoc(), newUnaryOutputType, newUnaryOpResult, perm);

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
    int32_t newAxis = invPerm[oldAxis];

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
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == newAxis) {
          resultShape[dim] += shape[dim];
        } else if (resultShape[dim] != shape[dim]) {
          // Dimensions must be equal except for the concatenation axis
          return failure();
        }
      }
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

struct HoistTransposeAbovePadPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input to TransposeOp is a PadOp
    auto padOp = dyn_cast_or_null<TFL::PadOp>(op.getInput().getDefiningOp());
    if (!padOp) {
      return failure();
    }

    // Get the permutation attribute
    DenseIntElementsAttr permAttr;
    if (!matchPattern(op.getPerm(), m_Constant(&permAttr))) {
      return failure();
    }
    auto perm = permAttr.getValues<int32_t>();

    // Get the padding attribute
    DenseIntElementsAttr padAttr;
    if (!matchPattern(padOp.getPadding(), m_Constant(&padAttr))) {
      return failure();
    }
    auto padValues = padAttr.getValues<int32_t>();

    // Get the rank of the tensor
    auto padInputType = padOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (!padInputType) {
      return failure();
    }
    int64_t rank = padInputType.getRank();

    // Reshape the padding values into a matrix of shape [rank, 2]
    SmallVector<int32_t, 8> paddingMatrix;
    paddingMatrix.reserve(padValues.size());
    for (int64_t i = 0; i < padValues.size(); ++i) {
      paddingMatrix.push_back(padValues[i]);
    }

    // Create a mapping from old dimensions to new dimensions after transpose
    SmallVector<int32_t, 8> inversePerm(rank);
    for (int64_t i = 0; i < rank; ++i) {
      inversePerm[perm[i]] = i;
    }

    // Permute the padding according to the inverse permutation
    SmallVector<int32_t, 8> newPaddingValues;
    newPaddingValues.reserve(paddingMatrix.size());
    for (int64_t i = 0; i < rank; ++i) {
      int32_t dim = inversePerm[i];
      newPaddingValues.push_back(paddingMatrix[dim * 2]);
      newPaddingValues.push_back(paddingMatrix[dim * 2 + 1]);
    }

    // Create new TransposeOp before PadOp's input
    auto newTransposeType = padOp.getInput().getType();
    auto newTranspose = rewriter.create<TFL::TransposeOp>(
        padOp.getLoc(), newTransposeType, padOp.getInput(), op.getPerm());

    // Create new padding constant
    auto newPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank, 2}, rewriter.getI32Type()),
        newPaddingValues);
    auto newPaddingConst = rewriter.create<arith::ConstantOp>(
        padOp.getLoc(), newPaddingAttr.getType(), newPaddingAttr);

    // Create new PadOp after TransposeOp
    auto newPadType = op.getType();
    auto newPad = rewriter.create<TFL::PadOp>(padOp.getLoc(), newPadType,
                                              newTranspose, newPaddingConst);

    rewriter.replaceOp(op, newPad.getResult());
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

        // Set function type to the transpose output type as we are changing the
        // input
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

  patterns.insert<HoistTransposeAbovePadPattern>(ctx);
  patterns.insert<FoldCancellableTransposePattern>(ctx);
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
