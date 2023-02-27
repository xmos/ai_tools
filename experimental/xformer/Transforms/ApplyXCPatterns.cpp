// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "lib_nn/api/nn_layers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Apply generated XC patterns.
struct ApplyXCPatterns
    : public PassWrapper<ApplyXCPatterns, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyXCPatterns)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-apply-xcpatterns"; }
  StringRef getDescription() const final {
    return "Apply generated XC optimization patterns.";
  }
  void runOnOperation() override;
};

StringAttr getPaddingPlan(PatternRewriter &rewriter, TFL::PadOp padOp) {
  DenseIntElementsAttr paddingAttr;
  if (!matchPattern(padOp.padding(), m_Constant(&paddingAttr))) {
    padOp.emitError("Could not obtain padding values.");
  }
  // Struct designated initializers not supported on Windows yet for C++17
  // Hence not used here
  padding_sizes_t paddingSizes = {
      /*.top = */ paddingAttr.getValues<int32_t>()[{1, 0}],
      /*.bottom = */ paddingAttr.getValues<int32_t>()[{1, 1}],
      /*.left = */ paddingAttr.getValues<int32_t>()[{2, 0}],
      /*.right = */ paddingAttr.getValues<int32_t>()[{2, 1}],
  };
  auto inputType =
      padOp.input().getType().template dyn_cast<RankedTensorType>();
  nn_image_params_t imageParams = {
      /*.height = */ static_cast<uint32_t>(inputType.getDimSize(1)),
      /*.width = */ static_cast<uint32_t>(inputType.getDimSize(2)),
      /*.channels = */ static_cast<channel_count_t>(inputType.getDimSize(3)),
  };

  nn_pad_plan_t paddingPlan;
  pad_prepare(&paddingPlan, &paddingSizes, &imageParams, imageParams.channels);
  auto paddingPlanData = std::string((char *)&paddingPlan, sizeof(paddingPlan));

  return rewriter.getStringAttr(paddingPlanData);
}

IntegerAttr getPadValue(PatternRewriter &rewriter, Value inputVal,
                        bool dontPack = false) {
  auto inputType = inputVal.getType().cast<ShapedType>();
  auto elementType = inputType.getElementType();

  // For quantized input type, padValue is the zero_point
  // Otherwise, it is zero
  int padValue = 0;
  if (elementType.isa<quant::QuantizedType>()) {
    auto inputQType = elementType.dyn_cast<quant::UniformQuantizedType>();
    padValue = inputQType.getZeroPoint();
    elementType = elementType.cast<quant::QuantizedType>().getStorageType();
  }

  if (!dontPack) {
    // padValue has to be four bytes
    // For int8, this would be arranged as b,b,b,b
    if (elementType.isInteger(8)) {
      padValue = padValue << 24 | (padValue << 16 & 0x00FFFFFF) |
                 (padValue << 8 & 0x0000FFFF) | (padValue & 0x000000FF);
    }
  } else {
    // TODO: Temp fix, need to pad negative values
    if (padValue < 0) {
      padValue = -padValue;
    }
  }

  return rewriter.getI32IntegerAttr(padValue);
}

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
  double inputScale = inputQType.getScale();
  int64_t inputZeroPoint = inputQType.getZeroPoint();

  // Get output scale and output zero point
  RankedTensorType outputType =
      op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  double outputScale = outputQType.getScale();
  assert(outputScale != 0 && "Output scale of zero is not supported!");
  int64_t outputZeroPoint = outputQType.getZeroPoint();

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
        return static_cast<int8_t>(std::max(
            {std::min({(int32_t)t, (int32_t)INT8_MAX}), (int32_t)INT8_MIN}));
      });

  ShapedType lookupTableType =
      RankedTensorType::get({256}, rewriter.getIntegerType(8));
  auto lookupTableAttr =
      DenseElementsAttr::get<int8_t>(lookupTableType, resultVector);
  return lookupTableAttr;
}

#include "Transforms/GeneratedXCPatterns.inc"

void ApplyXCPatterns::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  func::FuncOp func = getOperation();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyXCPatterns pass.
std::unique_ptr<OperationPass<func::FuncOp>> createApplyXCPatternsPass() {
  return std::make_unique<ApplyXCPatterns>();
}

static PassRegistration<ApplyXCPatterns> pass;

} // namespace xcore
} // namespace mlir
