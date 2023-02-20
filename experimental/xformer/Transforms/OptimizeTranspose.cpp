// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir {
namespace xcore {

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

struct HoistTransposeWCHAbovePadPattern
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Defining op must be pad
    auto padOp = dyn_cast_or_null<TFL::PadOp>(op.input().getDefiningOp());
    if (!padOp) {
      return failure();
    }

    // Get transpose permutation
    DenseIntElementsAttr perm;
    if (!matchPattern(op.perm(), m_Constant(&perm))) {
      return failure();
    }

    // Confirm transpose permutation is 0,2,3,1 i.e., NWCH
    // Remnants of Pytorch to TFlite conversion
    auto permVal = perm.getValues<int32_t>();
    if (permVal[0] != 0 || permVal[1] != 2 || permVal[2] != 3 ||
        permVal[3] != 1) {
      return failure();
    }

    // Get padding val
    DenseIntElementsAttr pad;
    if (!matchPattern(padOp.padding(), m_Constant(&pad))) {
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
        padOp.input().getType().cast<RankedTensorType>().getShape();
    auto tranposeResultType = RankedTensorType::get(
        {padInputShape[0], padInputShape[2], padInputShape[3],
         padInputShape[1]},
        padOp.input().getType().cast<ShapedType>().getElementType());
    auto newTranspose = rewriter.create<TFL::TransposeOp>(
        padOp.getLoc(), tranposeResultType, padOp.input(), op.perm());

    // Create new padding attr with spatial dimensions
    std::vector<int32_t> paddingValues{0, 0, 1, 1, 1, 1, 0, 0};
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4, 2}, rewriter.getI32Type()), paddingValues);
    auto paddingOp = rewriter.create<arith::ConstantOp>(
        padOp->getLoc(), RankedTensorType::get({4, 2}, rewriter.getI32Type()),
        paddingAttr);
    auto newPad = rewriter.create<TFL::PadOp>(
        padOp.getLoc(), op.output().getType(), newTranspose, paddingOp);

    rewriter.replaceOp(op, newPad.output());
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
        dyn_cast_or_null<TFL::TransposeOp>(op.input().getDefiningOp());
    if (!transposeOp) {
      return failure();
    }

    // Get transpose permutations
    DenseIntElementsAttr perm0;
    DenseIntElementsAttr perm1;
    if (!matchPattern(op.perm(), m_Constant(&perm0)) ||
        !matchPattern(transposeOp.perm(), m_Constant(&perm1))) {
      return failure();
    }

    // Do permutation indices cancel each other?
    if (!TF::AreCancellablePermutations(perm0, perm1)) {
      return failure();
    }

    rewriter.replaceOp(op, transposeOp.input());

    return success();
  }
};
struct FoldTransposeWCHToInput : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Defining op must be block arg?

    int k = 0;

    if (auto block_arg = op.input().dyn_cast<BlockArgument>()) {
      auto funcOp = cast<func::FuncOp>(block_arg.getOwner()->getParentOp());
      FunctionType func_type = funcOp.getFunctionType();
      llvm::SmallVector<Type, 4> new_input_types(func_type.getInputs().begin(),
                                                 func_type.getInputs().end());
      new_input_types[block_arg.getArgNumber()] = op.output().getType();
      auto newFuncType = FunctionType::get(
          rewriter.getContext(), new_input_types, funcOp.getResultTypes());
      funcOp.setType(newFuncType);

      block_arg.setType(op.output().getType());

      rewriter.replaceOp(op, op.input());
    }

    return success();
  }
};

void OptimizeTranspose::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);

  patterns.insert<HoistTransposeWCHAbovePadPattern>(ctx);
  patterns.insert<FoldCancellableTransposePattern>(ctx);
  patterns.insert<FoldTransposeWCHToInput>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OptimizeTranspose pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeTransposePass() {
  return std::make_unique<OptimizeTranspose>();
}

static PassRegistration<OptimizeTranspose> pass;

} // namespace xcore
} // namespace mlir
