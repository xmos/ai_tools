// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Apply generated patterns.
struct ApplyPatterns : public PassWrapper<ApplyPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

DenseElementsAttr getLookupTable(PatternRewriter &rewriter, Operation *op) {
  llvm::SmallVector<int8_t, 0> inputVector;
  inputVector.resize(256);

  // 0 -> 127
  // -128 -> -1
  std::iota(inputVector.begin(), inputVector.begin() + 128, 0);
  std::iota(inputVector.begin() + 128, inputVector.end(), -128);

  for (auto &i : inputVector)
    llvm::errs() << (int)i << "\n";

  llvm::errs() << "\n";

  // Get input scale
  RankedTensorType inputType =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  auto inputQType =
      inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto inputScale = inputQType.getScale();
  auto inputZeroPoint = inputQType.getZeroPoint();

  // Get output scale
  RankedTensorType outputType =
      op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto outputScale = outputQType.getScale();
  auto outputZeroPoint = outputQType.getZeroPoint();

  llvm::SmallVector<double, 0> dequantizedVector;
  std::transform(inputVector.begin(), inputVector.end(),
                 std::back_inserter(dequantizedVector), [&](int8_t n) {
                   return static_cast<double>(
                       (static_cast<int32_t>(n) - inputZeroPoint) * inputScale);
                 });

  for (auto &i : dequantizedVector)
    llvm::errs() << i << "\n";

  llvm::errs() << "\n";

  if (isa<TFL::ReluOp>(op)) {

    std::for_each(dequantizedVector.begin(), dequantizedVector.end(),
                  [](double &x) { x = std::max(x, 0.0); });

    for (auto &i : dequantizedVector)
      llvm::errs() << i << "\n";

    llvm::errs() << "\n";

    // array of -128 to 128 of int8
    // dequantize the array to f32 and find the activation function output
    // to dequantize, we need input scale and zero point

    // quantize the output array to get back to int8
    // to quantize, we need output scale and zero point
  }
  // t = np.round(np.float32(arr) / np.float32(scale)).astype(np.int32) +
  // zero_point return np.clip(t, np.iinfo(dtype).min,
  // np.iinfo(dtype).max).astype(dtype)

  llvm::SmallVector<int8_t, 0> resultVector;
  std::transform(
      dequantizedVector.begin(), dequantizedVector.end(),
      std::back_inserter(resultVector), [&](double n) {
        int32_t t =
            static_cast<int32_t>(round(n / outputScale)) + outputZeroPoint;
        return static_cast<int8_t>(std::max(std::min(t, INT8_MAX), INT8_MIN));
      });

  for (auto &i : resultVector)
    llvm::errs() << (int)i << "\n";

  ShapedType newWeightType =
      RankedTensorType::get({256}, rewriter.getIntegerType(8));
  auto newWeightAttr =
      DenseElementsAttr::get<int8_t>(newWeightType, resultVector);
  return newWeightAttr;
}

#include "Transforms/GeneratedPatterns.inc"

void ApplyPatterns::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  populateWithGenerated(ctx, patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}
} // namespace

// Creates an instance of the ApplyPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyPatternsPass() {
  return std::make_unique<ApplyPatterns>();
}

static PassRegistration<ApplyPatterns>
    pass("xcore-apply-patterns", "Apply generated optimization patterns.");

} // namespace xcore
} // namespace mlir
