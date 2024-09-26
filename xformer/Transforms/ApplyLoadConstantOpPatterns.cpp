// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
// Apply generated patterns.
struct ApplyLoadConstantOpPatterns
    : public PassWrapper<ApplyLoadConstantOpPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyLoadConstantOpPatterns)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final {
    return "xcore-apply-loadconstantop-patterns";
  }
  StringRef getDescription() const final {
    return "Apply load constant op optimization patterns.";
  }
  void runOnOperation() override;
};

static int totalSize_ = 0;

bool shouldBeLoadedExternally(Attribute values) {
  if (totalSize_ > maxLoadExternalSizeOption) {
    return false;
  }
  // values might be UnitAttr or BoolAttr which are too small to be loaded
  // externally anyway
  auto totalSizeInBits = 0;
  if (values.isa<DenseElementsAttr>()) {
    auto valuesAttr = values.cast<DenseElementsAttr>();
    totalSizeInBits =
        (valuesAttr.getNumElements() *
         valuesAttr.getType().getElementType().getIntOrFloatBitWidth());
  }
  totalSize_ += totalSizeInBits / CHAR_BIT;
  return totalSizeInBits / CHAR_BIT > loadExternallyIfLargerOption;
}

bool isNotUsedByLoadConstantOp(Value result) {
  if (result.hasOneUse()) {
    if (llvm::isa<LoadConstantOp>(*result.getUsers().begin())) {
      return false;
    }
  }
  return true;
}

#include "Transforms/GeneratedLoadConstantOpPatterns.inc"

void ApplyLoadConstantOpPatterns::runOnOperation() {
  func::FuncOp f = getOperation();
  if (weightsFilenameOption.empty()) {
    f.emitError("Weights file option should be provided to run this pass!");
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(&getContext());
  func::FuncOp func = getOperation();
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyLoadConstantOpPatterns pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createApplyLoadConstantOpPatternsPass() {
  return std::make_unique<ApplyLoadConstantOpPatterns>();
}

static PassRegistration<ApplyLoadConstantOpPatterns> pass;

} // namespace mlir::xcore
