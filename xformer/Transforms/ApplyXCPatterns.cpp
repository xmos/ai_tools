// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Utils/Diagnostics.h"
#include "Utils/Util.h"

#include "Transforms/Options.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "lib_nn/api/add_int16_transform.h"
#include "lib_nn/api/dequantize_int16_transform.h"
#include "lib_nn/api/multiply_int16_transform.h"
#include "lib_nn/api/nn_layers.h"
#include "lib_nn/api/quadratic_approximation.h"
#include "lib_nn/api/quantize_int16_transform.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include <numeric>

namespace mlir::xcore {

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

bool isBetaFloatEnabled() { return enableBetaFloatOption; }

IntegerAttr getPadValue(PatternRewriter &rewriter, Value inputVal) {
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

  assert(elementType.isIntOrFloat() &&
         "Type has to be I32, F32, or I8 if quantized!");
  // padValue has to be four bytes
  // For input type of int8, this would be arranged as b,b,b,b
  if (elementType.isInteger(8)) {
    padValue = padValue << 24 | (padValue << 16 & 0x00FFFFFF) |
               (padValue << 8 & 0x0000FFFF) | (padValue & 0x000000FF);
  }

  return rewriter.getI32IntegerAttr(padValue);
}

IntegerAttr getActivationType(PatternRewriter &rewriter, Operation *op) {
  // TODO: Refactor to use shared header file for enum
  if (isa<TFL::EluOp>(op)) {
    return rewriter.getI32IntegerAttr(0);
  } else if (isa<TFL::LogisticOp>(op)) {
    return rewriter.getI32IntegerAttr(1);
  } else if (isa<TFL::TanhOp>(op)) {
    return rewriter.getI32IntegerAttr(2);
  } else {
    llvm_unreachable("Unsupported op!");
  }
}

DenseElementsAttr getExpLookupF32(PatternRewriter &rewriter, Operation *op) {
  RankedTensorType inputType =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  auto inputQType =
      inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  double inputScale = inputQType.getScale();
  int64_t inputZeroPoint = inputQType.getZeroPoint();

  RankedTensorType outputType =
      op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  double outputScale = outputQType.getScale();
  int64_t outputZeroPoint = outputQType.getZeroPoint();
  assert(outputZeroPoint == -128 && outputScale == 1.0f / 256.0f &&
         "Output range must be 0-1");
  llvm::SmallVector<float, 0> resultVector;
  resultVector.resize(256);
  // generateExpLUT(inputZeroPoint, inputScale, resultVector.data());
  for (int i = 0; i < 256; i++) {
    float x = ((i - 128) - inputZeroPoint) * inputScale;
    resultVector[i] = expf(x);
  }
  ShapedType lookupTableType =
      RankedTensorType::get({256}, rewriter.getF32Type());
  auto lookupTableAttr =
      DenseElementsAttr::get<float>(lookupTableType, resultVector);
  return lookupTableAttr;
}

DenseElementsAttr getLookupTableI16(PatternRewriter &rewriter,
                                    Operation *activationOp, Operation *inputOp,
                                    Operation *outputOp) {
  // Get input scale and input zero point
  RankedTensorType inputType =
      inputOp->getOperand(0).getType().dyn_cast<RankedTensorType>();
  auto inputQType =
      inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  double inputScale = inputQType.getScale();
  int64_t inputZeroPoint = inputQType.getZeroPoint();

  // Get output scale and output zero point
  RankedTensorType outputType =
      outputOp->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  double outputScale = outputQType.getScale();
  assert(outputScale != 0 && "Output scale of zero is not supported!");
  int64_t outputZeroPoint = outputQType.getZeroPoint();

  float_function_t fn;

  if (isa<TFL::TanhOp>(activationOp)) {
    fn = approximation_function_tanh;
  } else if (isa<TFL::LogisticOp>(activationOp)) {
    fn = approximation_function_logistics;
  } else if (isa<TFL::EluOp>(activationOp)) {
    fn = approximation_function_elu;
  } else if (isa<TFL::ReluOp>(activationOp)) {
    fn = approximation_function_relu;
  } else {
    llvm_unreachable("Unsupported op!");
  }

  double square_error;
  int max_error;
  int chunks = 128;
  quadratic_function_table_t table;
  quadratic_approximation_generator(&table, fn, inputScale, outputScale, chunks,
                                    &max_error, &square_error);
  if (max_error > quadraticLookupErrorOption) {
    std::stringstream msg;
    msg << "Quadratic approximation error of " << max_error
        << " larger than set threshold of " << quadraticLookupErrorOption
        << ", therefore reverting to reference op!" << std::endl
        << "Inspect the output, and if suitable, set a "
           "higher threshold with --xcore-quadratic-lookup-error."
        << std::endl;
    activationOp->emitWarning(
        utils::getMsgWithLocPrefix(*activationOp, msg.str()));
    (void)rewriter.notifyMatchFailure(
        activationOp->getLoc(), "Cannot calculate quadratic approximation!");
    return {};
  }

  auto length = quadratic_function_table_number_bytes(&table);
  uint8_t *bytes = quadratic_function_table_bytes(&table);

  ArrayRef<uint8_t> tableData = ArrayRef(bytes, length);
  ShapedType lookupTableType = RankedTensorType::get(
      {length}, rewriter.getIntegerType(8, /*signed=*/false));
  auto lookupTableAttr =
      DenseElementsAttr::get<uint8_t>(lookupTableType, tableData);
  return lookupTableAttr;
}

DenseElementsAttr getLookupTableI16(PatternRewriter &rewriter, Operation *op) {
  return getLookupTableI16(rewriter, op, op, op);
}

DenseElementsAttr getLookupTableI8(PatternRewriter &rewriter, Operation *op) {
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
  } else if (isa<TFL::HardSwishOp>(op)) {
    std::for_each(
        dequantizedVector.begin(), dequantizedVector.end(),
        [](double &x) { x = x * std::min(std::max(x + 3, 0.0), 6.0) / 6; });
  } else {
    llvm_unreachable("Unsupported op!");
  }

  // Quantize to create the result vector
  llvm::SmallVector<uint8_t, 0> resultVector;
  std::transform(
      dequantizedVector.begin(), dequantizedVector.end(),
      std::back_inserter(resultVector), [&](double n) {
        int32_t t =
            static_cast<int32_t>(round(n / outputScale)) + outputZeroPoint;
        return static_cast<uint8_t>(std::max(
            {std::min({(int32_t)t, (int32_t)INT8_MAX}), (int32_t)INT8_MIN}));
      });

  ShapedType lookupTableType = RankedTensorType::get(
      {256}, rewriter.getIntegerType(8, /*signed=*/false));
  auto lookupTableAttr =
      DenseElementsAttr::get<uint8_t>(lookupTableType, resultVector);
  return lookupTableAttr;
}

DenseElementsAttr getBinaryI16Blob(PatternRewriter &rewriter, Operation *op,
                                   bool binaryInputs = true) {
  // Get input scale
  RankedTensorType inputType =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  double inputScale = 1;
  if (auto inputQType = inputType.getElementType()
                            .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    inputScale = inputQType.getScale();
  }

  double inputScale2 = 1;
  if (binaryInputs) {
    RankedTensorType inputType2 =
        op->getOperand(1).getType().dyn_cast<RankedTensorType>();
    if (auto inputQType2 = inputType2.getElementType()
                               .dyn_cast<mlir::quant::UniformQuantizedType>()) {
      inputScale2 = inputQType2.getScale();
    }
  }

  // Get output scale
  RankedTensorType outputType =
      op->getResult(0).getType().dyn_cast<RankedTensorType>();
  double outputScale = 1;
  if (auto outputQType = outputType.getElementType()
                             .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    outputScale = outputQType.getScale();
  }
  assert(outputScale != 0 && "Output scale of zero is not supported!");

  int length;
  std::vector<uint8_t> blob;
  std::string errMsg(ERR_MSG_DESCRIPTOR_FAIL_BYTES(), '\0');
  int succeeded;
  if (isa<TFL::QuantizeOp>(op) && inputType.getElementType().isF32()) {
    length = QUANTIZE_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded = quantize_int16_tensor_blob((void *)blob.data(), outputScale);
  } else if (isa<TFL::QuantizeOp>(op)) {
    length = REQUANTIZE_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded = requantize_int16_tensor_blob((void *)blob.data(), inputScale,
                                             outputScale, errMsg.data());
  } else if (isa<TFL::DequantizeOp>(op)) {
    length = DEQUANTIZE_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded = dequantize_int16_tensor_blob((void *)blob.data(), inputScale,
                                             errMsg.data());
  } else if (isa<TFL::AddOp>(op)) {
    length = ADD_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded = add_int16_tensor_blob((void *)blob.data(), inputScale,
                                      inputScale2, outputScale, errMsg.data());
  } else if (isa<TFL::SubOp>(op)) {
    length = ADD_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded = add_int16_tensor_blob((void *)blob.data(), inputScale,
                                      -inputScale2, outputScale, errMsg.data());
  } else if (isa<TFL::MulOp>(op)) {
    length = MULTIPLY_INT16_TENSOR_BYTES();
    blob.resize(length);
    succeeded =
        multiply_int16_tensor_blob((void *)blob.data(), inputScale, inputScale2,
                                   outputScale, errMsg.data());
  } else {
    llvm_unreachable("Unsupported op!");
  }

  if (!succeeded) {
    op->emitWarning(utils::getMsgWithLocPrefix(*op, errMsg));
    (void)rewriter.notifyMatchFailure(op->getLoc(), "Cannot obtain blob!");
    return {};
  }

  ArrayRef<uint8_t> blobData = ArrayRef(blob.data(), length);
  ShapedType blobType = RankedTensorType::get(
      {length}, rewriter.getIntegerType(8, /*signed=*/false));
  auto blobAttr = DenseElementsAttr::get<uint8_t>(blobType, blobData);
  return blobAttr;
}

DenseElementsAttr getUnaryI16Blob(PatternRewriter &rewriter, Operation *op) {
  return getBinaryI16Blob(rewriter, op, /*binaryInputs=*/false);
}

void calculateThreadSplit(int &tc, int split_size, int split_start[],
                          int split_end[]) {
  split_start[0] = 0;

  // Alignment is to four
  // Figure out min number of threads needed while keeping alignment
  // By dividing split_size by four and ceil that
  tc = std::min(tc, (split_size + 3) >> 2);

  for (int i = 0; i < tc; i++) {
    auto split = (split_size + (tc - i) - 1) / (tc - i);
    split_size -= split;
    if (split > 0) {
      split_end[i] = split_start[i] + split;
      if (i != tc - 1)
        split_start[i + 1] = split_end[i];
    } else {
      break;
    }
  }

  // Align up or down split_starts to word length = 4 bytes,
  // so that each thread begins work at an aligned address
  // The last thread handles remaining items, so don't modify the end
  for (int i = 1; i < tc; i++) {
    if ((split_start[i] & 3) >= 3) {
      // Align up
      split_start[i] = (split_start[i] + 3) & ~3;
    } else {
      // Align down
      split_start[i] = split_start[i] & ~3;
    }
    split_end[i - 1] = split_start[i];
  }
}

SmallVector<Value, 2> getBlobsForBlobUnaryI16(PatternRewriter &rewriter,
                                              Operation *op) {

  // TensorBlob, OperatorBlob, ThreadBlob, Input1, Input2, ...
  // Output1, Output2, ...

  // getUnaryI16InputTensorsSize
  // getUnaryI16OutputTensorsSize
  // getUnaryI16OperatorBlob

  // Get input scale
  RankedTensorType inputType =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  double inputScale = 1;
  if (auto inputQType = inputType.getElementType()
                            .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    inputScale = inputQType.getScale();
  }

  std::vector<uint8_t> blob;
  // Adding op type to the beginning
  int opBlobSize = DEQUANTIZE_INT16_TENSOR_BYTES() + 1;
  blob.resize(opBlobSize);
  blob[0] = 2;
  std::string errMsg(ERR_MSG_DESCRIPTOR_FAIL_BYTES(), '\0');
  dequantize_int16_tensor_blob((void *)((uint8_t*)blob.data() + 1), inputScale, errMsg.data());

  RankedTensorType type = RankedTensorType::get(
      {opBlobSize}, rewriter.getIntegerType(8, /*signed=*/false));
  auto attr = DenseIntElementsAttr::get(type, blob);
  auto opBlob = rewriter.create<arith::ConstantOp>(op->getLoc(), type, attr);

  // Upto 11 integers for five threads
  // First integer for number of threads used
  // Then start1, start2, ..., count1, count2, ...
  std::vector<int> thBlob;
  int s[5], e[5];
  int actualthreadCount = threadCountOption;
  calculateThreadSplit(actualthreadCount,
                       utils::getShapedTypeSize(inputType) /
                           utils::getTypeSize(inputType.getElementType()),
                       s, e);
  int threadBlobSize = actualthreadCount * 2 + 1;
  thBlob.resize(threadBlobSize);
  int thIndex = 0;
  thBlob[thIndex++] = actualthreadCount;
  for (int i = 0; i < actualthreadCount; i = i + 1) {
    thBlob[thIndex++] = s[i];
  }
  for (int i = 0; i < actualthreadCount; i = i + 1) {
    thBlob[thIndex++] = e[i] - s[i];
  }

  auto thType = RankedTensorType::get(
      {threadBlobSize * 4}, rewriter.getIntegerType(8, /*signed=*/false));
  auto thAttr = DenseElementsAttr::get<uint8_t>(
      thType, ArrayRef((uint8_t *)thBlob.data(), threadBlobSize * 4));
  auto threadBlob =
      rewriter.create<arith::ConstantOp>(op->getLoc(), thType, thAttr);

  return SmallVector<Value, 2>({opBlob, threadBlob});
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

} // namespace mlir::xcore
