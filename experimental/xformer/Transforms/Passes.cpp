// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Passes.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace xcore {

void buildXCorePassPipeline(OpPassManager &pm) {
  pm.addPass(createApplyTFLPatternsPass());
  pm.addPass(createReplaceAvgPoolWithConv2DPass());
  pm.addPass(createReplaceFCWithConv2DPass());
  pm.addPass(createPad3to4Conv2DPass());
  pm.addPass(createReplaceWithConv2DV2Pass());
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
