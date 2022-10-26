// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Passes.h"
#include "Transforms/Options.h"

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace xcore {

void buildXCorePassPipeline(OpPassManager &pm) {
  // Run pass from LCE to convert Larq ops which are in TFL custom op format to
  // Larq dialect
  pm.addPass(mlir::TFL::CreateTranslateToLCEPass());
  // TFL passes
  pm.addPass(createReplaceAvgPoolWithConv2DPass());
  pm.addPass(createReplaceFCWithConv2DPass());
  pm.addPass(createPad3to4Conv2DPass());
  pm.addPass(createApplyTFLPatternsPass());
  // XC passes
  pm.addPass(createInsertStridedSlicePatternsPass());
  pm.addPass(createInsertConcatPass());
  pm.addPass(createReplaceStridedSlicePass());
  pm.addPass(createReplaceConv2DPass());
  pm.addPass(createApplyXCPatternsPass());
  // Add to pipeline only if flash image file option is provided
  if (!flashImageFilenameOption.empty()) {
    pm.addPass(createApplyLoadConstantOpPatternsPass());
    pm.addPass(createWriteFlashImagePass());
  }
  // Run canonicalization, which includes combining Reshapes
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createTranslateToCustomOpPass());
}

void registerXCorePassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "xcore-tfl-pipeline",
      "Run XCore passes for transforming TFLite code into XCore",
      [](OpPassManager &passManager) { buildXCorePassPipeline(passManager); });
}

} // namespace xcore
} // namespace mlir
