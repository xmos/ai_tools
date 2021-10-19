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
struct ApplyLoadOpPatterns
    : public PassWrapper<ApplyLoadOpPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

#include "Transforms/GeneratedLoadOpPatterns.inc"

void ApplyLoadOpPatterns::runOnFunction() {
  assert(!flashImageFilenameOption.empty() &&
         "Flash image file option should be provided to run this pass!");

  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyLoadOpPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyLoadOpPatternsPass() {
  return std::make_unique<ApplyLoadOpPatterns>();
}

static PassRegistration<ApplyLoadOpPatterns>
    pass("xcore-apply-loadop-patterns", "Apply load op optimization patterns.");

} // namespace xcore
} // namespace mlir
