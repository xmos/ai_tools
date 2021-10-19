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
struct LoadOpPatterns : public PassWrapper<LoadOpPatterns, FunctionPass> {
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

#include "Transforms/GeneratedLoadOpPatterns.inc"

void LoadOpPatterns::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the LoadOpPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createLoadOpPatternsPass() {
  return std::make_unique<LoadOpPatterns>();
}

static PassRegistration<LoadOpPatterns>
    pass("xcore-apply-loadop-patterns", "Apply load op optimization patterns.");

} // namespace xcore
} // namespace mlir
