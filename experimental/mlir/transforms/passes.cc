#include "passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace xcore {

void buildXCorePassPipeline(OpPassManager &pm) {
  pm.addPass(mlir::xcore::createPrintNestingPass());
}

void registerXCorePassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "tfl-to-xcore-pipeline",
      "Run XCore passes for transforming TFLite code into XCore",
      [](OpPassManager &passManager) { buildXCorePassPipeline(passManager); });
}

} // namespace xcore
} // namespace mlir
