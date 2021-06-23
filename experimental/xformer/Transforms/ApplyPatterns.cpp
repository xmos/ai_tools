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

  // The inputvector has 256 input values in the following order,
  // 0, 1, 2... -> 127 and
  // -128, -127, -126... -> -1
  std::iota(inputVector.begin(), inputVector.begin() + 128, 0);
  std::iota(inputVector.begin() + 128, inputVector.end(), -128);

  // Get input scale and input zero point
  RankedTensorType inputType =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  auto inputQType =
      inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto inputScale = inputQType.getScale();
  auto inputZeroPoint = inputQType.getZeroPoint();

  // Get output scale and output zero point
  RankedTensorType outputType =
      op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto outputScale = outputQType.getScale();
  assert(outputScale != 0 && "Output scale of zero is not supported!");
  auto outputZeroPoint = outputQType.getZeroPoint();

  // Dequantize the input vector
  llvm::SmallVector<double, 0> dequantizedVector;
  std::transform(inputVector.begin(), inputVector.end(),
                 std::back_inserter(dequantizedVector), [&](int8_t n) {
                   return static_cast<double>(
                       (static_cast<int32_t>(n) - inputZeroPoint) * inputScale);
                 });

  // Apply the activation function to the dequantized vector
  if (isa<TFL::ReluOp>(op)) {
    std::for_each(dequantizedVector.begin(), dequantizedVector.end(),
                  [](double &x) { x = std::max(x, 0.0); });
  } else if (isa<TFL::Relu6Op>(op)) {
    std::for_each(dequantizedVector.begin(), dequantizedVector.end(),
                  [](double &x) { x = std::min(std::max(x, 0.0), 6.0); });
  } else if (isa<TFL::TanhOp>(op)) {
    std::for_each(dequantizedVector.begin(), dequantizedVector.end(),
                  [](double &x) { x = tanh(x); });
  } else if (isa<TFL::LogisticOp>(op)) {
    std::for_each(dequantizedVector.begin(), dequantizedVector.end(),
                  [](double &x) { x = 1.0 / (1.0 + exp(-x)); });
  } else {
    llvm_unreachable("Unsupported op!");
  }

  // Quantize to create the result vector
  llvm::SmallVector<int8_t, 0> resultVector;
  std::transform(
      dequantizedVector.begin(), dequantizedVector.end(),
      std::back_inserter(resultVector), [&](double n) {
        int32_t t =
            static_cast<int32_t>(round(n / outputScale)) + outputZeroPoint;
        return static_cast<int8_t>(std::max(std::min(t, INT8_MAX), INT8_MIN));
      });

  ShapedType lookupTableType =
      RankedTensorType::get({256}, rewriter.getIntegerType(8));
  auto lookupTableAttr =
      DenseElementsAttr::get<int8_t>(lookupTableType, resultVector);
  return lookupTableAttr;
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
