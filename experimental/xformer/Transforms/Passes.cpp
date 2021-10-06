// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace xcore {

void buildXCorePassPipeline(OpPassManager &pm) {
  pm.addPass(createApplyPatternsPass());
  pm.addPass(createReplaceFCWithConv2DPass());
  pm.addPass(createPad3to4Conv2DPass());
  pm.addPass(createReplaceWithConv2DV2Pass());
  pm.addPass(createLegalizeFullyConnectedPass());
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
