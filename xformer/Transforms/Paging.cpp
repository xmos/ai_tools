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

struct RaiseSliceLoadTensorInputPattern
    : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto f = slice->getParentOfType<func::FuncOp>();
    // If slice does not have a defining op, return failure
    if (!slice.getInput().getDefiningOp() ||
        !isa<LoadTensorOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    auto opOriginal =
        llvm::cast<LoadTensorOp>(slice.getInput().getDefiningOp());

    DenseElementsAttr beginAttr, sizeAttr;
    if (!matchPattern(slice.getBegin(), m_Constant(&beginAttr))) {
      return failure();
    }
    if (!matchPattern(slice.getSize(), m_Constant(&sizeAttr))) {
      return failure();
    }

    auto sliceOutShape = utils::getValShape(slice.getOutput());
    auto opReplacement = llvm::cast<LoadTensorOp>(rewriter.clone(*opOriginal));
    RankedTensorType opReplacementType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(opOriginal.getResult()));
    opReplacement->getResult(0).setType(opReplacementType);

    auto outputType =
        opOriginal.getResult().getType().template cast<RankedTensorType>();

    // replace slice with new slice -> new op
    rewriter.replaceOp(slice, opReplacement.getResult());

    return success();
  }
};

void Paging::runOnOperation() {
  auto func = getOperation();
  auto module = func->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();
  OpBuilder builder(func);

  auto &mem = getAnalysis<MemoryPlan>();
  llvm::StringMap<Value> inputTensorMap, outputTensorMap;
  mem.buildInputOutputTensorMaps(inputTensorMap, outputTensorMap);

  int address = 0;
  if (loadInputExternallyOption.size() > 0) {
    llvm::DenseSet<int> inputArgIndexSet;
    for (int i = 0; i < loadInputExternallyOption.size(); i = i + 1) {
      for (int j = 0; j < func.getNumArguments(); j++) {
        if (inputTensorMap[loadInputExternallyOption[i]] ==
            func.getArgument(j)) {
          inputArgIndexSet.insert(j);
        }
      }
    }
    std::vector<int> externalInputTensorsData;
    module->setAttr(kMetadataXCNumExternalInputTensors,
                    builder.getI32IntegerAttr(inputArgIndexSet.size()));

    for (auto index : inputArgIndexSet) {
      BlockArgument inp = func.getArgument(index);
      builder.setInsertionPointToStart(inp.getOwner());
      auto noValueOp = builder.create<TFL::NoValueOp>(
          inp.getLoc(), builder.getNoneType(), builder.getUnitAttr());
      llvm::SmallVector<Value> ops;
      ops.push_back(noValueOp);
      int size = utils::getShapedTypeSize(inp.getType().dyn_cast<ShapedType>());
      auto loadOp = builder.create<LoadTensorOp>(inp.getLoc(), inp.getType(),
                                                 ops, address, size);
      inp.replaceAllUsesWith(loadOp);

      // TODO
      externalInputTensorsData.push_back(index);
      externalInputTensorsData.push_back(address);
      externalInputTensorsData.push_back(size);

      // TODO
      address += size;
    }
    assert(externalInputTensorsData.size() == inputArgIndexSet.size() * 3);
    if (externalInputTensorsData.size()) {
      module->setAttr(kMetadataXCNumExternalInputTensorsData,
                      builder.getI32VectorAttr(externalInputTensorsData));
    }
  }

  if (storeOutputExternallyOption.size() > 0) {
    llvm::DenseSet<int> outputIndexSet;
    for (int i = 0; i < storeOutputExternallyOption.size(); i = i + 1) {
      auto term = func.back().getTerminator();
      for (int j = 0; j < term->getNumOperands(); j++) {
        if (outputTensorMap[storeOutputExternallyOption[i]] ==
            term->getOperand(j)) {
          outputIndexSet.insert(j);
        }
      }
    }
    std::vector<int> externalOutputTensorsData;
    module->setAttr(kMetadataXCNumExternalOutputTensors,
                    builder.getI32IntegerAttr(outputIndexSet.size()));

    auto term = func.back().getTerminator();
    for (auto index : outputIndexSet) {
      auto op = term->getOperand(index);
      builder.setInsertionPointAfterValue(op);

      int size = utils::getShapedTypeSize(op.getType().dyn_cast<ShapedType>());
      auto storeOp = builder.create<StoreTensorOp>(op.getLoc(), op.getType(),
                                                   op, address, size);
      term->setOperand(index, storeOp);

      // TODO
      externalOutputTensorsData.push_back(index);
      externalOutputTensorsData.push_back(address);
      externalOutputTensorsData.push_back(size);

      // TODO
      address += size;
    }

    // TOD
    assert(externalOutputTensorsData.size() == outputIndexSet.size() * 3);
    if (externalOutputTensorsData.size()) {
      module->setAttr(kMetadataXCNumExternalOutputTensorsData,
                      builder.getI32VectorAttr(externalOutputTensorsData));
    }
  }

  RewritePatternSet patterns1(ctx);
  patterns1.insert<CombineLoadSliceToPartialLoadPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns1));

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

  // Insert store_tensor and load_tensor based on liveness range for paging
  getAnalysisManager().clear();
  auto &m = getAnalysis<MemoryPlan>();
  auto opIdMap = m.getOperationsIDMap();
  auto ops = m.getOperationsSequence();
  auto values = m.getValuesSequence();
  auto vInfoMap = m.getValuesInfoMap();

  llvm::SetVector<int> convOpIds;

  for (auto v : values) {
    // If v is not constant
    // If first used and last used is more than livenessPagingOption
    // Go through all uses of value
    // Insert store tensor after value creation and then load tensor before each
    // use
    if (!vInfoMap[v].isConstant &&
        vInfoMap[v].lastUsed - vInfoMap[v].firstUsed > livenessPagingOption) {

      // DenseMap<OpOperand*, Type> opTypeMap;
      SmallVector<OpOperand *> uses;
      for (mlir::OpOperand &use : v.getUses()) {
        uses.push_back(&use);
      }

      auto dummyResultType =
          RankedTensorType::get({1}, builder.getIntegerType(8));

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
  llvm::outs() << "\nExternal memory size : " << address << "\n";
  module->setAttr("xc.paging_size", builder.getI32IntegerAttr(address));
  RewritePatternSet patterns5(ctx);
  patterns5.insert<CombineLoadSliceToPartialLoadPattern>(ctx);
  // patterns5.insert<CombineSliceStoreToPartialStorePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns5));
}
} // namespace

// Creates an instance of the Paging pass.
std::unique_ptr<OperationPass<func::FuncOp>> createPagingPass() {
  return std::make_unique<Paging>();
}

static PassRegistration<Paging> pass;

} // namespace mlir::xcore