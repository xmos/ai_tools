// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Apply generated TFL patterns.
struct ApplyTFLPatterns : public PassWrapper<ApplyTFLPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-apply-tflpatterns"; }
  StringRef getDescription() const final {
    return "Apply generated TFL optimization patterns";
  }
  void runOnFunction() override;
};

#include "Transforms/GeneratedTFLPatterns.inc"

void ApplyTFLPatterns::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyTFLPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyTFLPatternsPass() {
  return std::make_unique<ApplyTFLPatterns>();
}

static PassRegistration<ApplyTFLPatterns> pass;

} // namespace xcore
} // namespace mlir
