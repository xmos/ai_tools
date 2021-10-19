// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

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
  auto *ctx = &getContext();
  auto func = getFunction();

  std::string error_msg;
  auto output = mlir::openOutputFile("test.flash", &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return;
  }

  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto paramsVec = fbb.StartVector("params");

  OwningRewritePatternList patterns(ctx);
  patterns.insert<WriteFlashImagePattern>(&fbb, ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  fbb.EndVector(paramsVec, false, false);
  fbb.EndMap(rootMap);
  fbb.Finish();
  std::string options_bytes(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  output->os() << options_bytes;
  output->keep();
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
