// This transformation pass takes operations in TensorFlow dialect and
// optimizes them to resulting operations in TensorFlow.js dialect.

//#include <memory>

//#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"    // from @llvm-project
#include "mlir/IR/Matchers.h"      // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h" // from @llvm-project
#include "mlir/Pass/Pass.h"        // from @llvm-project
#include "mlir/Support/LLVM.h"     // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
//#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "../ir/xc_ops.h"

namespace mlir {
namespace xcore {

namespace {

// Optimize FullyConnected operations.
struct OptimizeFullyConnected
    : public PassWrapper<OptimizeFullyConnected, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

#include "experimental/mlir/transforms/generated_patterns.inc"

void OptimizeFullyConnected::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  populateWithGenerated(ctx, patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}
} // namespace

// Creates an instance of the OptimizeFullyConnected pass.
std::unique_ptr<OperationPass<FuncOp>> createOptimizeFullyConnectedPass() {
  return std::make_unique<OptimizeFullyConnected>();
}

static PassRegistration<OptimizeFullyConnected>
    pass("xcore-optimize-fullyconnected",
         "Optimize FullyConnected operations.");

} // namespace xcore
} // namespace mlir
