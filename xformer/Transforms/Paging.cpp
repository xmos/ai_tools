// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
struct Paging : public PassWrapper<Paging, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Paging)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-run-paging"; }
  StringRef getDescription() const final { return "Run paging pass"; }
  void runOnOperation() override;
};

struct ConvertToStoreLoadPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto f = slice->getParentOfType<func::FuncOp>();

    // TODO
    // if (!slice->hasAttr(opSplitLabel) ||
    //     slice->hasAttr(opSplitLabelNumSplits)) {
    //   return failure();
    // }

    printf("found slice\n");

    auto dummyResultType =
        RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto sliceReplacement = rewriter.create<StoreTensorOp>(
        slice.getLoc(), dummyResultType, slice.getInput(), 1, 1);

    SmallVector<Value> sliceOps;
    sliceOps.push_back(sliceReplacement);
    auto loadOp = rewriter.create<LoadTensorOp>(
        slice.getLoc(), slice.getOutput().getType(), sliceOps, 1, 1);

    // replace slice with new slice -> new pad
    rewriter.replaceOp(slice, loadOp.getOutput());

    return success();
  }
};

struct ConvertToStoreLoadConcatPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat,
                                PatternRewriter &rewriter) const override {
    // TODO
    // if (!concat->hasAttr(opSplitLabel) ||
    //     concat->hasAttr(opSplitLabelNumSplits)) {
    //   return failure();
    // }

    printf("found concat\n");

    SmallVector<Value> storeOps;
    for (int i = 0; i < concat->getNumOperands(); ++i) {
      auto dummyResultType =
          RankedTensorType::get({1}, rewriter.getIntegerType(8));
      rewriter.setInsertionPointAfter(concat.getOperand(i).getDefiningOp());
      auto storeOp = rewriter.create<StoreTensorOp>(
          concat.getLoc(), dummyResultType, concat.getOperand(i), 1, 1);
      storeOps.push_back(storeOp);
    }
    auto loadOp = rewriter.create<LoadTensorOp>(
        concat.getLoc(), concat.getOutput().getType(), storeOps, 1, 1);

    // replace slice with new slice -> new pad
    rewriter.replaceOp(concat, loadOp.getOutput());

    return success();
  }
};

struct CombineLoadSliceToPartialLoadPattern
    : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    if (!(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (!isa<LoadTensorOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    auto loadOriginal =
        llvm::cast<LoadTensorOp>(slice.getInput().getDefiningOp());
    int addressOriginal = loadOriginal.getAddress();

    auto loadReplacement =
        llvm::cast<LoadTensorOp>(rewriter.clone(*loadOriginal));

    DenseElementsAttr attr;
    if (!matchPattern(slice.getBegin(), m_Constant(&attr))) {
      return failure();
    }

    // Calculate strides
    auto inputShape =
        slice.getInput().getType().dyn_cast<ShapedType>().getShape();
    int batchStride = inputShape[1] * inputShape[2] * inputShape[3];
    int heightStride = inputShape[2] * inputShape[3];
    int widthStride = inputShape[3];
    int channelStride = 1;

    auto beginVal = attr.getValues<int32_t>();
    int addressOffset = beginVal[0] * batchStride + beginVal[1] * heightStride +
                        beginVal[2] * widthStride + beginVal[3] * channelStride;
    addressOffset *=
        utils::getTypeSize(slice.getInput().getType().getElementType());

    int newSize = utils::getShapedTypeSize(
        slice.getOutput().getType().dyn_cast<ShapedType>());

    loadReplacement.setAddress(addressOriginal + addressOffset);
    loadReplacement.setSize(newSize);
    loadReplacement->getResult(0).setType(slice.getOutput().getType());
    rewriter.replaceOp(slice, loadReplacement.getOutput());

    return success();
  }
};

// struct CombineSliceStoreToPartialStorePattern : public
// OpRewritePattern<StoreTensorOp> {
//   using OpRewritePattern<StoreTensorOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(StoreTensorOp store,
//                                 PatternRewriter &rewriter) const override {
//     if (!(store.getInput().getDefiningOp())) {
//       return failure();
//     }

//     if (!isa<TFL::SliceOp>(store.getInput().getDefiningOp())) {
//       return failure();
//     }

//     auto slice = store.getInput().getDefiningOp();

//     store->getResult(0).setType(slice->getResult(0).getType());
//     store->setOperand(0, slice->getOperand(0));
//     rewriter.replaceOp(slice, store->getResult(0));

//     return success();
//   }
// };

struct ReorderLoadStorePattern : public OpRewritePattern<StoreTensorOp> {
  using OpRewritePattern<StoreTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreTensorOp storeTensorOp,
                                PatternRewriter &rewriter) const override {
    auto opInput = storeTensorOp.getInput();

    if (opInput.getDefiningOp() && isa<LoadTensorOp>(opInput.getDefiningOp())) {
      storeTensorOp->moveAfter(opInput.getDefiningOp());
      return success();
    }

    return failure();
  }
};

void Paging::runOnOperation() {
  auto func = getOperation();
  auto *ctx = &getContext();
  OpBuilder builder(func);

  // TODO
  // Try store_tensor and load_tensor ops
  // Op split slices are lowered to one store, and multiple partial loads
  // Op split concats are lowered to partial stores, and one load
  // RewritePatternSet patterns3(ctx);
  // patterns3.insert<ConvertToStoreLoadPattern>(ctx);
  // patterns3.insert<ConvertToStoreLoadConcatPattern>(ctx);
  // (void)applyPatternsAndFoldGreedily(func, std::move(patterns3));

  // RewritePatternSet patterns4(ctx);
  // patterns4.insert<CombineStoresPattern>(ctx);
  // (void)applyPatternsAndFoldGreedily(func, std::move(patterns4));

  // RewritePatternSet patterns5(ctx);
  // patterns5.insert<ReorderLoadStorePattern>(ctx);
  // (void)applyPatternsAndFoldGreedily(func, std::move(patterns5));

  // Reorder async load to be before previous convolution
  // so that the compute can be overlapped with the load
  auto &m = getAnalysis<MemoryPlan>();
  auto opIdMap = m.getOperationsIDMap();
  auto ops = m.getOperationsSequence();
  auto values = m.getValuesSequence();
  auto vInfoMap = m.getValuesInfoMap();

  llvm::SetVector<int> convOpIds;

  int address = 0;

  for (auto v : values) {
    // if v is not constant
    // if first used and last used if more than ten
    // go through all uses of value
    // insert store tensor after value creation and then load tensor before each
    // use
    if (!vInfoMap[v].isConstant &&
        vInfoMap[v].lastUsed - vInfoMap[v].firstUsed > livenessPagingOption) {

      // DenseMap<OpOperand*, Type> opTypeMap;
      SmallVector<OpOperand *> uses;
      for (mlir::OpOperand &use : v.getUses()) {
        // opTypeMap[&use] = use.get().getType();
        uses.push_back(&use);
      }

      auto dummyResultType =
          RankedTensorType::get({1}, builder.getIntegerType(8));

      // SmallVector<Value> storeOps;
      Value storeOp;
      int size = utils::getShapedTypeSize(v.getType().dyn_cast<ShapedType>());

      if (auto blockArg = v.dyn_cast<BlockArgument>()) {
        builder.setInsertionPointToStart(blockArg.getOwner());
        storeOp = builder.create<StoreTensorOp>(v.getLoc(), dummyResultType, v,
                                                address, size);
      } else {
        Operation *defOp = v.getDefiningOp();
        builder.setInsertionPointAfter(defOp);
        storeOp = builder.create<StoreTensorOp>(
            defOp->getLoc(), dummyResultType, v, address, size);
      }

      for (OpOperand *use : uses) {
        mlir::Operation *op = use->getOwner();
        builder.setInsertionPoint(op);
        auto loadOp = builder.create<LoadTensorOp>(op->getLoc(), v.getType(),
                                                   storeOp, address, size);
        use->set(loadOp.getResult());
      }

      address += size;
    }
  }
  printf("\nDDR size = %d", address);
  RewritePatternSet patterns5(ctx);
  patterns5.insert<CombineLoadSliceToPartialLoadPattern>(ctx);
  // patterns5.insert<CombineSliceStoreToPartialStorePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns5));

  // move input to ddr
  // add pass to add one load tensor at input
  // if load tensor is immediately followed by a store tensor of the same size
  // other load tensors, remove the first load tensor
}
} // namespace

// Creates an instance of the Paging pass.
std::unique_ptr<OperationPass<func::FuncOp>> createPagingPass() {
  return std::make_unique<Paging>();
}

static PassRegistration<Paging> pass;

} // namespace mlir::xcore