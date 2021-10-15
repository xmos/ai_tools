// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Apply generated patterns.
struct ApplyPatterns2 : public PassWrapper<ApplyPatterns2, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

bool isFlashImageFileProvided() {
  if (flashImageFilenameOption.empty()) {
    return false;
  }
  return true;
}

#include "Transforms/GeneratedPatterns2.inc"

void ApplyPatterns2::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyPatterns2 pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyPatterns2Pass() {
  return std::make_unique<ApplyPatterns2>();
}

static PassRegistration<ApplyPatterns2>
    pass("xcore-apply-patterns2", "Apply generated optimization patterns2.");

} // namespace xcore
} // namespace mlir
