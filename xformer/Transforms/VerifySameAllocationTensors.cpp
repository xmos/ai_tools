// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
struct VerifySameAllocationTensors
    : public PassWrapper<VerifySameAllocationTensors,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySameAllocationTensors)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TFLDialect>();
  }
  StringRef getArgument() const final { return "xcore-preset-allocations"; }
  StringRef getDescription() const final { return "Remove dynamic shape"; }
  void runOnOperation() override;
};

void VerifySameAllocationTensors::runOnOperation() {
  auto func = getOperation();
  auto *ctx = &getContext();

  if (sameAllocationInputOutputTensorOption.size() > 0) {

    auto &m = getAnalysis<MemoryPlan>();
    llvm::StringMap<Value> inputTensorMap, outputTensorMap;
    m.buildInputOutputTensorMaps(inputTensorMap, outputTensorMap);

    bool failed = false;
    // Check names of input and output tensors
    for (int i = 0; i < sameAllocationInputOutputTensorOption.size();
         i = i + 2) {
      if (!inputTensorMap.count(sameAllocationInputOutputTensorOption[i])) {
        func.emitError()
            << sameAllocationInputOutputTensorOption[i]
            << " not present in input tensors. Please check the name!";
        failed = true;
      }
      if (!outputTensorMap.count(
              sameAllocationInputOutputTensorOption[i + 1])) {
        func.emitError()
            << sameAllocationInputOutputTensorOption[i + 1]
            << " not present in output tensors. Please check the name!";
        failed = true;
      }
    }

    if (failed) {
      signalPassFailure();
      return;
    }

    // Check sizes
    auto vInfo = m.getValuesInfoMap();
    for (int i = 0; i < sameAllocationInputOutputTensorOption.size();
         i = i + 2) {
      if (vInfo[inputTensorMap[sameAllocationInputOutputTensorOption[i]]]
              .size !=
          vInfo[outputTensorMap[sameAllocationInputOutputTensorOption[i + 1]]]
              .size) {
        func.emitError() << "Size of input tensor "
                         << sameAllocationInputOutputTensorOption[i]
                         << " is not equal to output tensor "
                         << sameAllocationInputOutputTensorOption[i + 1]
                         << ". Please check!";
        failed = true;
      }
    }

    // Check quantization
    for (int i = 0; i < sameAllocationInputOutputTensorOption.size();
         i = i + 2) {
      auto inQType = dyn_cast_or_null<quant::UniformQuantizedType>(
          inputTensorMap[sameAllocationInputOutputTensorOption[i]]
              .getType()
              .cast<RankedTensorType>()
              .getElementType());
      auto outQType = dyn_cast_or_null<quant::UniformQuantizedType>(
          outputTensorMap[sameAllocationInputOutputTensorOption[i + 1]]
              .getType()
              .cast<RankedTensorType>()
              .getElementType());
      if (inQType && !outQType) {
        func.emitError() << "Input tensor "
                         << sameAllocationInputOutputTensorOption[i]
                         << " is quantized, but "
                         << sameAllocationInputOutputTensorOption[i + 1]
                         << " is not. Please check!";
        failed = true;
      } else if (!inQType && outQType) {
        func.emitError() << "Input tensor "
                         << sameAllocationInputOutputTensorOption[i]
                         << " is not quantized, but "
                         << sameAllocationInputOutputTensorOption[i + 1]
                         << " is quantized. Please check!";
        failed = true;
      } else if (inQType && outQType) {
        // Both are quantized, but check element sizes, maybe i8 and i16

        auto inScale = inQType.getScale();
        auto inZeroPoint = inQType.getZeroPoint();

        auto outScale = outQType.getScale();
        auto outZeroPoint = outQType.getZeroPoint();
        if (inScale != outScale || inZeroPoint != outZeroPoint) {
          // change input block arg to output quantization

          // insert quantize op to convert back to original input quantization
          // auto module = func->getParentOfType<ModuleOp>();
          // OpBuilder builder(module);
          // auto outVal =
          // outputTensorMap[sameAllocationInputOutputTensorOption[i
          // + 1]]; auto newQType = inQType.castFromExpressedType(
          //     quant::QuantizedType::castToExpressedType(outVal.getType()));
          // auto newQuantizeOp = builder.create<TFL::QuantizeOp>(
          //     inVal.getLoc(), newQType, outVal, TypeAttr::get(inQType));

          auto inVal = inputTensorMap[sameAllocationInputOutputTensorOption[i]];
          auto typeNumBits =
              utils::getTypeSize(
                  inVal.getType().cast<RankedTensorType>().getElementType()) *
              8;
          double maxError = 1.0 / (2 << (typeNumBits - 1));
          if (abs(inScale - outScale) > maxError) {
            func.emitError()
                << "Input tensor " << sameAllocationInputOutputTensorOption[i]
                << " has scale of " << inScale << " and zeropoint of "
                << inZeroPoint << ", but output tensor "
                << sameAllocationInputOutputTensorOption[i + 1]
                << " has scale of " << outScale << " and zeropoint of "
                << outZeroPoint << ". Please check!";
            failed = true;
          }
        }
      } else if (!inQType && !outQType) {
        // Both are not quantized, but check element sizes, maybe i8 and i16
      }
    }

    if (failed) {
      signalPassFailure();
      return;
    }
  }
}
} // namespace

// Creates an instance of the VerifySameAllocationTensors pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createVerifySameAllocationTensorsPass() {
  return std::make_unique<VerifySameAllocationTensors>();
}

static PassRegistration<VerifySameAllocationTensors> pass;

} // namespace mlir::xcore
