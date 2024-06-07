// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
// Write flash image
struct PlanMemory
    : public PassWrapper<PlanMemory, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanMemory)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-plan-memory"; }
  StringRef getDescription() const final { return "Plan memory"; }
  void runOnOperation() override;
};

void PlanMemory::runOnOperation() {
  if (offlineOffsetsOption) {
    auto func = getOperation();

    bool unSupportedOpsInGraph = false;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (llvm::isa<TFL::UnidirectionalSequenceLSTMOp, TFL::WhileOp, TFL::IfOp,
                    TFL::CallOnceOp>(op)) {
        unSupportedOpsInGraph = true;
      }
    });

    if (!unSupportedOpsInGraph) {
      auto module = func->getParentOfType<ModuleOp>();
      OpBuilder builder(module);

      auto &m = getAnalysis<MemoryPlan>();
      int peakMemoryUsedWithOverlap, peakMemoryUsedWithoutOverlap, peakOpId;
      auto offlineOffsetsWithOverlap = m.getAllocatedOffsets(
          /*overlapOps=*/true, peakMemoryUsedWithOverlap, peakOpId);
      auto offlineOffsetsWithoutOverlap = m.getAllocatedOffsets(
          /*overlapOps=*/false, peakMemoryUsedWithoutOverlap, peakOpId);
      module->setAttr("xc.peakopid", builder.getI32IntegerAttr(peakOpId));
      module->setAttr("xc.peakusage",
                      builder.getI32IntegerAttr(peakMemoryUsedWithoutOverlap));

      if (peakMemoryUsedWithOverlap <= peakMemoryUsedWithoutOverlap) {
        module->setAttr("xc.offsets",
                        builder.getI32VectorAttr(offlineOffsetsWithOverlap));
      } else {
        module->setAttr("xc.offsets",
                        builder.getI32VectorAttr(offlineOffsetsWithoutOverlap));
      }
    }
  }
}
} // namespace

// Creates an instance of the PlanMemory pass.
std::unique_ptr<OperationPass<func::FuncOp>> createPlanMemoryPass() {
  return std::make_unique<PlanMemory>();
}

static PassRegistration<PlanMemory> pass;

} // namespace mlir::xcore
