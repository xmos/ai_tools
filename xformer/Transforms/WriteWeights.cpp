// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/FileIO.h"
#include "Utils/TileRamSupport.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::xcore {

namespace {
// Write weights to a file
struct WriteWeights
    : public PassWrapper<WriteWeights, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WriteWeights)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-write-weights"; }
  StringRef getDescription() const final { return "Write weights"; }
  void runOnOperation() override;
};

struct WriteWeightsPattern : public OpRewritePattern<LoadConstantOp> {
  WriteWeightsPattern(std::vector<std::vector<char>> *tensorsVec,
                      MLIRContext *context)
      : OpRewritePattern<LoadConstantOp>(context), tensorsVec_(tensorsVec) {}

  std::vector<char> getTensorData(LoadConstantOp loadOp) const {
    DenseElementsAttr attr;
    if (loadOp.getInput()
            .getType()
            .cast<ShapedType>()
            .getElementType()
            .isa<quant::QuantizedType>()) {
      auto qConstOp =
          dyn_cast<TFL::QConstOp>(loadOp.getInput().getDefiningOp());
      attr = qConstOp.getValue().template cast<DenseElementsAttr>();
    } else {
      matchPattern(loadOp.getInput(), m_Constant(&attr));
    }

    std::vector<char> tensorData;
    int n = attr.isSplat() ? attr.getNumElements() : 1;
    for (int i = 0; i < n; ++i) {
      tensorData.insert(tensorData.end(), attr.getRawData().begin(),
                        attr.getRawData().end());
    }
    return tensorData;
  }

  LogicalResult matchAndRewrite(LoadConstantOp loadOp,
                                PatternRewriter &rewriter) const override {
    std::vector<char> tensorData;
    SmallVector<Attribute> dataSizes;

    int address = 0;
    for (auto const &t : *tensorsVec_) {
      address += t.size();
    }

    // We try to combine loads to one op if the load has only one use or if the
    // load is not from external memory.
    // External memory loads have to be aligned to 32 bytes/256 bits for max
    // speed
    LoadWeightsOpType opType = LoadWeightsOpType::Sync;
    if (loadOp.getResult().hasOneUse() && !weightsInExternalMemory) {
      auto use = loadOp->use_begin();
      Operation *ownerOp = use->getOwner();

      SmallVector<Type> outputTypes;
      SmallVector<int> opNums;

      for (int i = 0; i < ownerOp->getNumOperands(); i++) {
        auto loadOpForOwnerOp = dyn_cast_or_null<LoadConstantOp>(
            ownerOp->getOperand(i).getDefiningOp());

        if (loadOpForOwnerOp) {
          std::vector<char> loadOpData = getTensorData(loadOpForOwnerOp);
          dataSizes.push_back(rewriter.getI32IntegerAttr(loadOpData.size()));
          tensorData.insert(tensorData.end(), loadOpData.begin(),
                            loadOpData.end());
          outputTypes.push_back(loadOpForOwnerOp.getType());
          opNums.push_back(i);
        }
      }

      auto loadWeightsOp = rewriter.create<LoadWeightsOp>(
          loadOp.getLoc(), outputTypes, address,
          rewriter.getArrayAttr(dataSizes), stringifyLoadWeightsOpType(opType));

      for (int i = 0; i < opNums.size(); i++) {
        ownerOp->setOperand(opNums[i], loadWeightsOp.getResult(i));
      }

      loadWeightsOp->moveBefore(ownerOp);
      loadOp.erase();
    } else {
      std::vector<char> loadOpData = getTensorData(loadOp);
      dataSizes.push_back(rewriter.getI32IntegerAttr(loadOpData.size()));
      tensorData.insert(tensorData.end(), loadOpData.begin(), loadOpData.end());
      if (weightsInExternalMemory) {
        // Pad tensordata to 32 bytes alignment
        auto alignedSize = ((loadOpData.size() + 31) / 32) * 32;
        auto toBePaddedSize = alignedSize - loadOpData.size();
        // Pad with zeros
        tensorData.insert(tensorData.end(), toBePaddedSize, 0);
        opType = LoadWeightsOpType::DDR;
      }
      auto loadWeightsOp = rewriter.create<LoadWeightsOp>(
          loadOp.getLoc(), loadOp.getType(), address,
          rewriter.getArrayAttr(dataSizes), stringifyLoadWeightsOpType(opType));
      rewriter.replaceOp(loadOp, loadWeightsOp.getOutput());

      // Find all uses of loadWeightsOp and find the first Owner op
      // so that we can move the loading to just before that op.
      mlir::Operation *firstOwnerOp =
          loadWeightsOp->getResult(0).getUses().begin()->getOwner();
      for (const mlir::OpOperand &use : loadWeightsOp->getResult(0).getUses()) {
        mlir::Operation *op = use.getOwner();
        if (op->isBeforeInBlock(firstOwnerOp)) {
          firstOwnerOp = op;
        }
      }
      loadWeightsOp->moveBefore(firstOwnerOp);
    }

    tensorsVec_->push_back(tensorData);

    return success();
  }

private:
  std::vector<std::vector<char>> *tensorsVec_;
};

struct LowerToAsyncLoadsPattern : public OpRewritePattern<LoadWeightsOp> {
  LowerToAsyncLoadsPattern(MLIRContext *context)
      : OpRewritePattern<LoadWeightsOp>(context) {}

  LogicalResult matchAndRewrite(LoadWeightsOp loadWeightsOp,
                                PatternRewriter &rewriter) const override {
    if (loadWeightsOp.getOpType() !=
        stringifyLoadWeightsOpType(LoadWeightsOpType::Sync)) {
      return failure();
    }

    // We use loadWeightsOp.getResultTypes() as Load Weights op can have
    // variadic number of results
    auto loadWeightsAsyncOp = rewriter.create<LoadWeightsOp>(
        loadWeightsOp.getLoc(), loadWeightsOp.getResultTypes(),
        loadWeightsOp.getAddress(), loadWeightsOp.getSizes(),
        stringifyLoadWeightsOpType(LoadWeightsOpType::Async));

    auto loadWeightsWaitOp = rewriter.create<LoadWeightsWaitOp>(
        loadWeightsAsyncOp.getLoc(), loadWeightsAsyncOp.getResultTypes(),
        loadWeightsAsyncOp.getResults());

    rewriter.replaceOp(loadWeightsOp, loadWeightsWaitOp.getOutput());

    return success();
  }
};

void WriteWeights::runOnOperation() {
  func::FuncOp f = getOperation();
  if (weightsFilenameOption.empty()) {
    f.emitError("Weights file option should be provided to run this pass!");
    signalPassFailure();
    return;
  }

  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  // For each LoadOp in the graph, save the tensor data, and replace the LoadOp
  // with a LoadWeightsOp
  std::vector<std::vector<char>> tensorsVec;
  RewritePatternSet patterns(ctx);
  patterns.insert<WriteWeightsPattern>(&tensorsVec, ctx);
  if (asyncLoadWeightsOption) {
    patterns.insert<LowerToAsyncLoadsPattern>(ctx);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Reorder async load to be before previous convolution
  // so that the compute can be overlapped with the load
  auto &m = getAnalysis<MemoryPlan>();
  auto opIdMap = m.getOperationsIDMap();
  auto ops = m.getOperationsSequence();

  llvm::SetVector<int> convOpIds;
  for (auto o : ops) {
    if (llvm::isa<Conv2DV2Op>(o)) {
      convOpIds.insert(opIdMap[o]);
    }
  }

  for (auto o : ops) {
    if (llvm::isa<LoadWeightsOp>(o)) {
      auto ldOp = dyn_cast<LoadWeightsOp>(o);
      if (ldOp.getOpType() ==
          stringifyLoadWeightsOpType(LoadWeightsOpType::Async)) {
        int idx = llvm::lower_bound(convOpIds, opIdMap[o]) - convOpIds.begin();
        if (idx > 0) {
          o->moveBefore(ops[convOpIds[idx - 1]]);
        }
      }
    }
  }

  if (failed(utils::writeWeightsToFile(weightsFilenameOption, tensorsVec,
                                       weightsAsArrayOption,
                                       weightsInExternalMemory))) {
    f.emitError("Failed to write weights to file!");
    signalPassFailure();
    return;
  }
}
} // namespace

// Creates an instance of the WriteWeights pass.
std::unique_ptr<OperationPass<func::FuncOp>> createWriteWeightsPass() {
  return std::make_unique<WriteWeights>();
}

static PassRegistration<WriteWeights> pass;

} // namespace mlir::xcore
