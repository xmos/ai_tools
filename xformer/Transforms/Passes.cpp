// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Passes.h"
#include "Transforms/Options.h"

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::xcore {

void buildXCorePreOpSplitPassPipeline(OpPassManager &pm) {
  // Run pass from LCE to convert Larq ops which are in TFL custom op format to
  // Larq dialect
  pm.addPass(mlir::TFL::CreateTranslateToLCEPass());
  // Convert dynamic shapes in batch dimension to static
  pm.addPass(createRemoveDynamicShapePass());
}

void buildXCoreRemainingPassPipeline(OpPassManager &pm) {
  // TFL passes
  pm.addPass(createOptimizeTransposePass());
  pm.addPass(createReplaceAvgPoolWithConv2DPass());
  pm.addPass(createReplaceFCWithConv2DPass());
  if (opSplitTensorArenaOption) {
    pm.addPass(createOpSplitPass());
  }
  pm.addPass(createApplyTFLPatternsPass());
  pm.addPass(createReplaceAvgPoolWithConv2DPass());
  pm.addPass(createOptimizeConv2DPass());
  pm.addPass(createApplyTFLPatternsPass());
  pm.addPass(createReplaceStridedSlicePass());
  // Run canonicalization, which includes combining Reshapes
  pm.addPass(mlir::createCanonicalizerPass());

  // XC passes
  pm.addPass(createReplaceAddSubPass());
  pm.addPass(createReplaceMaxPoolPass());
  pm.addPass(createReplaceMulPass());
  pm.addPass(createReplaceTransposeConvPass());
  pm.addPass(createReplaceConv2DPass());
  pm.addPass(createReplacePadPass());
  pm.addPass(createReplaceSlicePass());
  pm.addPass(createReplaceBroadcastPass());
  pm.addPass(createReplaceConcatPass());
  pm.addPass(createApplyXCPatternsPass());
  // Add to pipeline only if weights file option is provided
  if (!weightsFilenameOption.empty()) {
    pm.addPass(createApplyLoadConstantOpPatternsPass());
    pm.addPass(createWriteWeightsPass());
  }
  // Run canonicalization, which includes combining Reshapes
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createPlanMemoryPass());
  pm.addPass(createTranslateToCustomOpPass());
}

void registerXCorePassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "xcore-tfl-pipeline",
      "Run XCore passes for transforming TFLite code into XCore",
      [](OpPassManager &passManager) {
        buildXCorePreOpSplitPassPipeline(passManager);
        buildXCoreRemainingPassPipeline(passManager);
      });
}

} // namespace mlir::xcore
