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
struct WriteFlashImage : public PassWrapper<WriteFlashImage, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

struct WriteFlashImagePattern : public OpRewritePattern<LoadOp> {
  WriteFlashImagePattern(std::vector<std::vector<char>> *tensorsVec,
                         MLIRContext *context)
      : OpRewritePattern<LoadOp>(context), tensorsVec_(tensorsVec) {}

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr attr;
    if (!matchPattern(loadOp.input(), m_Constant(&attr))) {
      return failure();
    }
    std::vector<char> tensorData = attr.getRawData().vec();

    // Create a LoadFlashOp with vector index and tensor size
    auto loadFlashOp =
        rewriter.create<LoadFlashOp>(loadOp.getLoc(), loadOp.getType(),
                                     tensorsVec_->size(), tensorData.size());
    tensorsVec_->push_back(tensorData);

    // Replace the LoadOp with the new LoadFlashOp
    rewriter.replaceOp(loadOp, loadFlashOp.output());
    return success();
  }

private:
  std::vector<std::vector<char>> *tensorsVec_;
};

void WriteFlashImage::runOnFunction() {
  assert(!flashImageFilenameOption.empty() &&
         "Flash image file option should be provided to run this pass!");

  auto *ctx = &getContext();
  auto func = getFunction();
  // For each LoadOp in the graph, save the tensor data, and replace the LoadOp
  // with a LoadFlashOp
  std::vector<std::vector<char>> tensorsVec;
  OwningRewritePatternList patterns(ctx);
  patterns.insert<WriteFlashImagePattern>(&tensorsVec, ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Write tensor data to flash image file
  if (failed(
          utils::writeFlashImageToFile(flashImageFilenameOption, tensorsVec))) {
    llvm::errs() << "Failed to write flash image!\n";
  }
}
} // namespace

// Creates an instance of the WriteFlashImage pass.
std::unique_ptr<OperationPass<FuncOp>> createWriteFlashImagePass() {
  return std::make_unique<WriteFlashImage>();
}

static PassRegistration<WriteFlashImage> pass("xcore-write-flash-image",
                                              "Write flash image.");

} // namespace xcore
} // namespace mlir
