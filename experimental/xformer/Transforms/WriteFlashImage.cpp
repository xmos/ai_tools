// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/FileIO.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace xcore {

namespace {
// Write flash image
struct WriteFlashImage
    : public PassWrapper<WriteFlashImage, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WriteFlashImage)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-write-flash-image"; }
  StringRef getDescription() const final { return "Write flash image"; }
  void runOnOperation() override;
};

struct WriteFlashImagePattern : public OpRewritePattern<LoadConstantOp> {
  WriteFlashImagePattern(std::vector<std::vector<char>> *tensorsVec,
                         MLIRContext *context)
      : OpRewritePattern<LoadConstantOp>(context), tensorsVec_(tensorsVec) {}

  std::vector<char> getTensorData(LoadConstantOp loadOp) const {
    DenseElementsAttr attr;
    if (loadOp.input()
            .getType()
            .cast<ShapedType>()
            .getElementType()
            .isa<quant::QuantizedType>()) {
      auto qConstOp = dyn_cast<TFL::QConstOp>(loadOp.input().getDefiningOp());
      attr = qConstOp.value().template cast<DenseElementsAttr>();
    } else {
      matchPattern(loadOp.input(), m_Constant(&attr));
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

    // Constants are usually allocated in the beginning of the function.
    // Lowering them to load from flash op leads to loading constants from flash
    // occurring in the beginning of graph execution before other ops are
    // executed, thereby needing a much larger tensor arena.
    // We move the op to right before the user op (user op would be conv or
    // lookup op etc, any op that is using the constant).
    // This is so that when we lower to flatbuffer the loadOp will be located
    // in the graph close to the user op.
    if (loadOp.getResult().hasOneUse()) {
      auto use = loadOp->use_begin();
      Operation *ownerOp = use->getOwner();
      loadOp->moveBefore(ownerOp);

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

      auto loadFlashOp =
          rewriter.create<LoadFlashOp>(loadOp.getLoc(), outputTypes, address,
                                       rewriter.getArrayAttr(dataSizes));

      for (int i = 0; i < opNums.size(); i++) {
        ownerOp->setOperand(opNums[i], loadFlashOp.getResult(i));
      }
      loadOp.erase();
    } else {
      std::vector<char> loadOpData = getTensorData(loadOp);
      dataSizes.push_back(rewriter.getI32IntegerAttr(loadOpData.size()));
      tensorData.insert(tensorData.end(), loadOpData.begin(), loadOpData.end());
      auto loadFlashOp = rewriter.create<LoadFlashOp>(
          loadOp.getLoc(), loadOp.getType(), address,
          rewriter.getArrayAttr(dataSizes));
      rewriter.replaceOp(loadOp, loadFlashOp.output());
    }

    tensorsVec_->push_back(tensorData);

    return success();
  }

private:
  std::vector<std::vector<char>> *tensorsVec_;
};

void WriteFlashImage::runOnOperation() {
  func::FuncOp f = getOperation();
  if (flashImageFilenameOption.empty()) {
    f.emitError("Flash image file option should be provided to run this pass!");
    signalPassFailure();
    return;
  }

  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  // For each LoadOp in the graph, save the tensor data, and replace the LoadOp
  // with a LoadFlashOp
  std::vector<std::vector<char>> tensorsVec;
  RewritePatternSet patterns(ctx);
  patterns.insert<WriteFlashImagePattern>(&tensorsVec, ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Write tensor data to flash image file
  if (failed(
          utils::writeFlashImageToFile(flashImageFilenameOption, tensorsVec))) {
    f.emitError("Failed to write flash image!");
    signalPassFailure();
    return;
  }
}
} // namespace

// Creates an instance of the WriteFlashImage pass.
std::unique_ptr<OperationPass<func::FuncOp>> createWriteFlashImagePass() {
  return std::make_unique<WriteFlashImage>();
}

static PassRegistration<WriteFlashImage> pass;

} // namespace xcore
} // namespace mlir
