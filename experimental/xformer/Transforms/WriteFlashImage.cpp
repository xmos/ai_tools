// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/FileIO.h"

#include "flatbuffers/flexbuffers.h"
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
  WriteFlashImagePattern(flexbuffers::Builder *fbb, MLIRContext *context)
      : OpRewritePattern<LoadOp>(context), fbb_(fbb) {}

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {

    // check that we can open a file
    // iterating through load op
    // get op data which would be a vector
    // getOffsetAndWriteToFlashImage(vector)
    // create an ld flash op with offset and tensor size

    std::vector<int8_t> weights = {1, 2, 3, 4, 5, 6, 7, 8};

    fbb_->Blob(weights.data(), weights.size());

    // write to file
    // get the offset
    // replace the load op with a load flash one

    auto loadFlashOp = rewriter.create<LoadFlashOp>(
        loadOp.getLoc(), loadOp.getType(), loadOp.input(), 0, 0);

    // Replace the FC with the new ops
    rewriter.replaceOp(loadOp, loadFlashOp.output());

    return success();
  }

private:
  flexbuffers::Builder *fbb_;
};

void WriteFlashImage::runOnFunction() {
  assert(!flashImageFilenameOption.empty() &&
         "Flash image file option should be provided to run this pass!");

  auto *ctx = &getContext();
  auto func = getFunction();

  // Create flexbuffer of params
  // For each LoadOp in the graph, we add a new flexbuffer blob
  // and replace the LoadOp with a LoadFlashOp
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto paramsVec = fbb.StartVector("params");

  OwningRewritePatternList patterns(ctx);
  patterns.insert<WriteFlashImagePattern>(&fbb, ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  fbb.EndVector(paramsVec, false, false);
  fbb.EndMap(rootMap);
  fbb.Finish();

  // Write flexbuffer data to flash image file
  std::string fbbData(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  if (failed(utils::writeDataToFile(flashImageFilenameOption, fbbData))) {
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
