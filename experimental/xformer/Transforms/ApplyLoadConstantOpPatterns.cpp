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
struct ApplyLoadConstantOpPatterns
    : public PassWrapper<ApplyLoadConstantOpPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final {
    return "xcore-apply-loadconstantop-patterns";
  }
  StringRef getDescription() const final {
    return "Apply load constant op optimization patterns";
  }
  void runOnFunction() override;
};

bool shouldBeLoadedExternally(Attribute values) {
  auto valuesAttr = values.cast<DenseElementsAttr>();
  auto totalSizeInBits = (valuesAttr.getNumElements() *
                          valuesAttr.getElementType().getIntOrFloatBitWidth());
  return totalSizeInBits / CHAR_BIT > loadExternallyIfLargerOption;
}

bool isUsedByValidOp(Value constOpType) {
  for (auto *operand : constOpType.getUsers()) {
    // In the runtime, TFL::PadOp assumes that the constant data is available in
    // the flatbuffer, and fails if it's not available
    if (llvm::isa<TFL::PadOp>(operand)) {
      return false;
    }
  }
  return true;
}

#include "Transforms/GeneratedLoadConstantOpPatterns.inc"

void ApplyLoadConstantOpPatterns::runOnFunction() {
  auto f = getFunction();
  if (flashImageFilenameOption.empty()) {
    f.emitError("Flash image file option should be provided to run this pass!");
    signalPassFailure();
    return;
  }

  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyLoadConstantOpPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyLoadConstantOpPatternsPass() {
  return std::make_unique<ApplyLoadConstantOpPatterns>();
}

static PassRegistration<ApplyLoadConstantOpPatterns> pass;

} // namespace xcore
} // namespace mlir
