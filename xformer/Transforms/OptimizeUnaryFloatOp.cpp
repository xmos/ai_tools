// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Options.h"

#include "Utils/Util.h"
#include "IR/XCoreOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir::xcore {

namespace {
// Optimize TFL Unary Float op.
struct OptimizeUnaryFloatOp
    : public PassWrapper<OptimizeUnaryFloatOp, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeUnaryFloatOp)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-optimize-unary-float-op"; }
  StringRef getDescription() const final { return "Optimize TFL unary float op."; }
  void runOnOperation() override;
};

void calculateOutputScaleAndZeroPoint(
  double foutput0, double foutput1, int64_t ioutput0, int64_t ioutput1,
  int64_t *outputZeroPoint, double *outputScale) {
  // Calculate the output scale and output zero point
  // zero point = (f0*n1-f1*n0)/(f0-f1)
  double f0 = foutput0;
  double f1 = foutput1;
  double n0 = static_cast<double>(ioutput0);
  double n1 = static_cast<double>(ioutput1);
  *outputZeroPoint = static_cast<int64_t>((f0*n1-f1*n0)/(f0-f1));
  // scale = f0 / (n0 - z)
  *outputScale = f0 / (n0 - static_cast<double>(*outputZeroPoint));
}

struct MoveDequantForwardAndReplaceSqrtPattern
    : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequantOp,
                                PatternRewriter &rewriter) const override {
    TFL::SqrtOp sqrtOp = nullptr;
    func::ReturnOp returnOp = nullptr;
    // Ensure the DequantizeOp has a single use, we can accept if other use is ReturnOp
    for (auto *user : dequantOp.getResult().getUsers()) {
      if (isa<TFL::SqrtOp>(user) && sqrtOp == nullptr)
        sqrtOp = dyn_cast<TFL::SqrtOp>(user);
      else if (isa<func::ReturnOp>(user) && returnOp == nullptr)
        returnOp = dyn_cast<func::ReturnOp>(user);
      else{
        return failure();
      }
    }

    if (sqrtOp == nullptr) return failure();

    if (returnOp) {
      // Verify that returnOp belongs to the entry function
      auto funcOp = returnOp->getParentOfType<func::FuncOp>();
      auto module = funcOp->getParentOfType<ModuleOp>();
      if (!funcOp || !module) return failure();
    }

    llvm::SmallVector<int8_t, 0> inputVector;
    inputVector.resize(256);

    // The inputvector has 256 input values in the following order,
    // 0, 1, 2... -> 127 and
    // -128, -127, -126... -> -1
    std::iota(inputVector.begin(), inputVector.begin() + 128, 0);
    std::iota(inputVector.begin() + 128, inputVector.end(), -128);

    // Get input scale and zero point
    RankedTensorType inputType =
        dequantOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputQType =
        inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    double inputScale = inputQType.getScale();
    int64_t inputZeroPoint = inputQType.getZeroPoint();

    // Dequantize the input vector
    llvm::SmallVector<double, 0> dequantizedVector;
    std::transform(
      inputVector.begin(), inputVector.end(), std::back_inserter(dequantizedVector), 
      [&](int8_t n) {
        return static_cast<double>(
            (static_cast<int32_t>(n) - inputZeroPoint) * inputScale);
      });

    // Apply sqrt function to the dequantized vector
    std::for_each(
      dequantizedVector.begin(), dequantizedVector.end(),
      [](double &x) { x = std::sqrt(x); });

    // Calculate the output scale and output zero point
    int64_t outputZeroPoint;
    double outputScale;
    calculateOutputScaleAndZeroPoint(
      dequantizedVector[127], dequantizedVector[128],
      inputVector[127], inputVector[128],
      &outputZeroPoint, &outputScale);

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
        
    // create arith constantop for lookup op here
    auto lookupConstOp = rewriter.create<arith::ConstantOp>(
      dequantOp.getLoc(), lookupTableAttr);

    // create lookup table op here
    UniformQuantizedType newSqrtResultQType = UniformQuantizedType::get(
      true, rewriter.getIntegerType(8), rewriter.getF32Type(),
      outputScale, outputZeroPoint, 
      QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
      QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

    auto newSqrtResultType = RankedTensorType::get(
        inputType.getShape(), newSqrtResultQType);

    auto newSqrtOp = rewriter.create<LookupOp>(
      dequantOp.getLoc(), newSqrtResultType, dequantOp.getInput(), lookupConstOp);

    // Create a new dequantize operation after the sqrt operation
    auto newDequantizeOp = rewriter.create<TFL::DequantizeOp>(
        dequantOp.getLoc(), sqrtOp.getResult().getType(), newSqrtOp.getResult());

    rewriter.replaceOp(dequantOp, newSqrtOp.getResult());
    rewriter.replaceOp(sqrtOp, newDequantizeOp.getResult());

    if (returnOp) {
      // Update function signature
      auto funcOp = returnOp->getParentOfType<func::FuncOp>();
      // Build new function type using the return operands (now quantized)
      auto newFuncType = FunctionType::get(
          getContext(),
          funcOp.getArgumentTypes(),
          llvm::to_vector<4>(returnOp->getOperandTypes())   // converts to SmallVector<Type>
      );
      funcOp.setType(newFuncType);
    }

    return success();
  }
};

struct MoveDequantForwardOverUnaryOpPattern
    : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequantOp,
                                PatternRewriter &rewriter) const override {
    Operation* userOp = nullptr;
    func::ReturnOp returnOp = nullptr;
    // Ensure the DequantizeOp has a single use, we can accept if other use is ReturnOp
    for (auto *user : dequantOp.getResult().getUsers()) {
      if (isa<TFL::AbsOp, TFL::NegOp, TFL::LogOp, TFL::LogisticOp, TFL::SumOp>(user) 
        && userOp == nullptr)
        userOp = user;
      else if (isa<func::ReturnOp>(user) && returnOp == nullptr)
        returnOp = dyn_cast<func::ReturnOp>(user);
      else{
        return failure();
      }
    }

    if (userOp == nullptr) return failure();

    if (returnOp) {
      // Verify that returnOp belongs to the entry function
      auto funcOp = returnOp->getParentOfType<func::FuncOp>();
      auto module = funcOp->getParentOfType<ModuleOp>();
      if (!funcOp || !module) return failure();
    }

    // Get input scale and zero point
    RankedTensorType inputType =
        dequantOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputQType =
        inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    double inputScale = inputQType.getScale();
    int64_t inputZeroPoint = inputQType.getZeroPoint();

    // Get the dequantized input
    int64_t qMin = QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8);
    int64_t qMax = QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8);
    double fInputMin = static_cast<double>((qMin - inputZeroPoint) * inputScale);
    double fInputMax = static_cast<double>((qMax - inputZeroPoint) * inputScale);

    Value newUnaryOpResult;
    auto loc = userOp->getLoc();
    auto input = dequantOp.getInput();

    // Retrieve the original unary operation's output type
    auto originalUnaryOutputType =
        userOp->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!originalUnaryOutputType)
      return failure();

    if (auto absOp = dyn_cast<TFL::AbsOp>(userOp)) {
      // Get the maximum floating point output of AbsOp
      double fOutputMax = std::max(std::abs(fInputMin), std::abs(fInputMax));
      // Get the minimum floating point output of AbsOp
      double fOutputMin = 0.0;
      if (qMax - inputZeroPoint < 0) {
        fOutputMin = fInputMax;
      }
      int64_t outputZeroPoint;
      double outputScale;
      calculateOutputScaleAndZeroPoint(
        fOutputMax, fOutputMin, qMax, qMin, &outputZeroPoint, &outputScale);

      UniformQuantizedType newAbsResultQType = UniformQuantizedType::get(
        true, rewriter.getIntegerType(8), rewriter.getF32Type(),
        outputScale, outputZeroPoint, 
        QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
        QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

      auto newAbsResultType = RankedTensorType::get(
          inputType.getShape(), newAbsResultQType, originalUnaryOutputType.getEncoding());

      newUnaryOpResult =
          rewriter.create<TFL::AbsOp>(loc, newAbsResultType, input);
    } else if (auto negOp = dyn_cast<TFL::NegOp>(userOp)) {
      // NegOp require input and output type the same
      newUnaryOpResult =
          rewriter.create<TFL::NegOp>(loc, inputType, input);
    } else if (auto logisticOp = dyn_cast<TFL::LogisticOp>(userOp)) {
      // LogisticOp require input and output type the same
      newUnaryOpResult =
          rewriter.create<TFL::LogisticOp>(loc, inputType, input);
    } else if (auto logOp = dyn_cast<TFL::LogOp>(userOp)) {
      // LogOp require input and output type the same
      newUnaryOpResult =
          rewriter.create<TFL::LogisticOp>(loc, inputType, input);
    } else if (auto sumOp = dyn_cast<TFL::SumOp>(userOp)) {
      auto axes = sumOp.getAxes();
      auto keepDim = sumOp.getKeepDimsAttr();

      UniformQuantizedType newSumResultQType = UniformQuantizedType::get(
        true, rewriter.getIntegerType(8), rewriter.getF32Type(),
        inputScale, inputZeroPoint, 
        QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
        QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

      auto newSumResultType = RankedTensorType::get(
          originalUnaryOutputType.getShape(),
          newSumResultQType, originalUnaryOutputType.getEncoding());

      newUnaryOpResult =
          rewriter.create<TFL::SumOp>(
            loc, newSumResultType, input, axes, keepDim);
    } else {
      // This should not happen as we checked the op type earlier
      return failure();
    }

    // Create a new Dequantize operation after the unary operation
    auto newDequantOp = rewriter.create<TFL::DequantizeOp>(
        dequantOp.getLoc(), originalUnaryOutputType, newUnaryOpResult);

    // Replace the original user operation's result with the new dequant
    // result
    rewriter.replaceOp(userOp, newDequantOp.getResult());

    // Remove the original dequantOp
    rewriter.eraseOp(dequantOp);

    return success();
  }
};

struct MoveDequantForwardOverSameInputOpPattern
    : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequantOp,
                                PatternRewriter &rewriter) const override {
    Operation* userOp = nullptr;
    func::ReturnOp returnOp = nullptr;
    // Ensure the DequantizeOp has a single use, we can accept if other use is ReturnOp
    for (auto *user : dequantOp.getResult().getUsers()) {
      if (isa<TFL::MulOp>(user) && (userOp == nullptr || userOp == user)) // TODO: add more qualify ops
        userOp = user;
      else if (isa<func::ReturnOp>(user) && returnOp == nullptr)
        returnOp = dyn_cast<func::ReturnOp>(user);
      else{
        return failure();
      }
    }

    if (userOp == nullptr) return failure();

    if (auto mulOp = dyn_cast<TFL::MulOp>(userOp)) {
      if (mulOp.getRhs() != mulOp.getLhs())
        return failure();
    }

    if (returnOp) {
      // Verify that returnOp belongs to the entry function
      auto funcOp = returnOp->getParentOfType<func::FuncOp>();
      auto module = funcOp->getParentOfType<ModuleOp>();
      if (!funcOp || !module) return failure();
    }

    // Get input scale and zero point
    RankedTensorType inputType =
        dequantOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputQType =
        inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    double inputScale = inputQType.getScale();
    int64_t inputZeroPoint = inputQType.getZeroPoint();

    // Get the dequantized input
    int64_t qMin = QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8);
    int64_t qMax = QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8);
    double fInputMin = static_cast<double>((qMin - inputZeroPoint) * inputScale);
    double fInputMax = static_cast<double>((qMax - inputZeroPoint) * inputScale);

    Value newOpResult;
    auto loc = userOp->getLoc();
    auto input = dequantOp.getInput();

    // Retrieve the original user operation's output type
    auto originalOutputType =
        userOp->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!originalOutputType)
      return failure();

    if (auto mulOp = dyn_cast<TFL::MulOp>(userOp)) {
      // Get the maximum floating point output of MulOp
      double fOutputMax = std::max(fInputMin*fInputMin, fInputMax*fInputMax);
      // Get the minimum floating point output of MulOp
      double fOutputMin = std::min(fInputMin*fInputMin, fInputMax*fInputMax);
      int64_t outputZeroPoint;
      double outputScale;
      calculateOutputScaleAndZeroPoint(
        fOutputMax, fOutputMin, qMax, qMin, &outputZeroPoint, &outputScale);

      UniformQuantizedType newMulResultQType = UniformQuantizedType::get(
        true, rewriter.getIntegerType(8), rewriter.getF32Type(),
        outputScale, outputZeroPoint, 
        QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
        QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

      auto mulOpOutputShape = mulOp.getResult().getType().getShape();
      auto newMulResultType = RankedTensorType::get(
          mulOpOutputShape, newMulResultQType, originalOutputType.getEncoding());

      newOpResult =
          rewriter.create<TFL::MulOp>(loc, newMulResultType,
            input, input, mulOp.getFusedActivationFunction());
    } else {
      // This should not happen as we checked the op type earlier
      return failure();
    }

    // Create a new Dequantize operation after the user operation
    auto newDequantOp = rewriter.create<TFL::DequantizeOp>(
        dequantOp.getLoc(), originalOutputType, newOpResult);

    // Replace the original user operation's result with the new dequant
    // result
    rewriter.replaceOp(userOp, newDequantOp.getResult());

    // Remove the original dequantOp
    rewriter.eraseOp(dequantOp);

    return success();
  }
};

// Fold Dequant -> Mul (rank0 const) -> Quant into Quant
struct FoldDequantMulQuantOpPattern
    : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequantOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the DequantizeOp has a single use
    if (!dequantOp->hasOneUse())
      return failure();
    // Ensure after DequantizeOp is MulOp
    TFL::MulOp mulOp = dyn_cast_or_null<TFL::MulOp>(*dequantOp->getUsers().begin());
    if (!mulOp)
      return failure();
    // Ensure MulOp didn't fuse with activation function
    if (mulOp.getFusedActivationFunction() != "NONE")
      return failure();
    // Ensure MulOp has a single use
    if (!mulOp->hasOneUse())
      return failure();
    // Ensure after MulOp is QuantizeOp
    TFL::QuantizeOp quantOp = dyn_cast_or_null<TFL::QuantizeOp>(*mulOp->getUsers().begin());
    if (!quantOp)
      return failure();

    auto rhs = mulOp.getRhs();
    auto rhsDefOp = rhs.getDefiningOp();
    if (!rhsDefOp) return failure();
    float multiplier = 0.0f;

    if (auto tflConst = dyn_cast<TFL::ConstOp>(rhsDefOp)) {
      auto denseAttr = tflConst.getValue().cast<DenseElementsAttr>();
      multiplier = denseAttr.getSplatValue<float>();
    } else if (auto arithConst = dyn_cast<arith::ConstantOp>(rhsDefOp)) {
      auto attr = arithConst.getValue();
      if (auto denseAttr = attr.dyn_cast<DenseElementsAttr>()) {
        multiplier = denseAttr.getSplatValue<float>();
      } else if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
        multiplier = floatAttr.getValueAsDouble();
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    // Get quantOp output scale and zero point
    RankedTensorType outputType =
        quantOp.getResult().getType().dyn_cast<RankedTensorType>();
    auto outputQType =
        outputType.getElementType().dyn_cast<quant::UniformQuantizedType>();
    double outputScale = outputQType.getScale();
    int64_t outputZeroPoint = outputQType.getZeroPoint();

    // Create a new Quantize operation
    UniformQuantizedType newQuantQType = UniformQuantizedType::get(
      true, rewriter.getIntegerType(8), rewriter.getF32Type(),
      multiplier/outputScale, outputZeroPoint, 
      QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
      QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

    auto newQuantResultType = RankedTensorType::get(
        outputType.getShape(), newQuantQType);

    auto newQuantOp = rewriter.create<TFL::QuantizeOp>(
        dequantOp.getLoc(), newQuantResultType, dequantOp.getInput(), 
        TypeAttr::get(newQuantResultType));

    rewriter.replaceOp(quantOp, newQuantOp.getResult());

    // Remove the original ops
    rewriter.eraseOp(mulOp);
    rewriter.eraseOp(dequantOp);

    return success();
  }
};

LookupOp CreateLookupOp(
  void (*func)(double&), Location loc,
  TypedValue<TensorType> input, TypedValue<TensorType> output, PatternRewriter &rewriter) {

  llvm::SmallVector<int8_t, 0> inputVector;
  inputVector.resize(256);

  // The inputvector has 256 input values in the following order,
  // 0, 1, 2... -> 127 and
  // -128, -127, -126... -> -1
  std::iota(inputVector.begin(), inputVector.begin() + 128, 0);
  std::iota(inputVector.begin() + 128, inputVector.end(), -128);

  // Get input scale and zero point
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  auto inputQType = inputType.getElementType().dyn_cast<quant::UniformQuantizedType>();
  double inputScale = inputQType.getScale();
  int64_t inputZeroPoint = inputQType.getZeroPoint();

  // Dequantize the input vector
  llvm::SmallVector<double, 0> dequantizedVector;
  std::transform(
    inputVector.begin(), inputVector.end(), std::back_inserter(dequantizedVector), 
    [&](int8_t n) {
      return static_cast<double>(
          (static_cast<int32_t>(n) - inputZeroPoint) * inputScale);
    });

  // Apply func to the dequantized vector
  std::for_each(
    dequantizedVector.begin(), dequantizedVector.end(),
    func);

  // Calculate the output scale and output zero point
  int64_t outputZeroPoint;
  double outputScale;
  calculateOutputScaleAndZeroPoint(
    dequantizedVector[127], dequantizedVector[128],
    inputVector[127], inputVector[128],
    &outputZeroPoint, &outputScale);

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
      
  // create arith constantop for lookup op here
  auto lookupConstOp = rewriter.create<arith::ConstantOp>(
    loc, lookupTableAttr);

  // create lookup table op here
  UniformQuantizedType newResultQType = UniformQuantizedType::get(
    true, rewriter.getIntegerType(8), rewriter.getF32Type(),
    outputScale, outputZeroPoint, 
    QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8),
    QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8));

  auto newResultType = RankedTensorType::get(
      inputType.getShape(), newResultQType);

  auto newOp = rewriter.create<LookupOp>(
    loc, newResultType, input, lookupConstOp);

  return newOp;
}

/**
 * Fold X -> Dequant \
 *                    => Div -> Quant -> Z
 *      Y -> Dequant /
 * into
 *                X \
 *                   => Mul -> Z
 *      Y -> Lookup /
 * Where Lookup ops turn Y into 1/Y 
 */ 
struct FoldDequantDivQuantOpPattern
    : public OpRewritePattern<TFL::DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DivOp divOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the DivOp lhs is a Dequantize Op
    TFL::DequantizeOp lhsDequantOp = 
      dyn_cast_or_null<TFL::DequantizeOp>(divOp.getLhs().getDefiningOp());
    if (!lhsDequantOp)
      return failure();
    // Ensure the DivOp rhs is a Dequantize Op
    TFL::DequantizeOp rhsDequantOp = 
      dyn_cast_or_null<TFL::DequantizeOp>(divOp.getRhs().getDefiningOp());
    if (!rhsDequantOp)
      return failure();
    // Ensure the DivOp has a single use
    if (!divOp->hasOneUse())
      return failure();
    // Ensure after DivOp is Quantize Op
    TFL::QuantizeOp quantOp = dyn_cast_or_null<TFL::QuantizeOp>(*divOp->getUsers().begin());
    if (!quantOp)
      return failure();

    auto lookupOp = CreateLookupOp(
      [](double &x) { x = 1.0/x; },
      rhsDequantOp.getLoc(), rhsDequantOp.getInput(), rhsDequantOp.getResult(), rewriter);
    
    auto mulOp = rewriter.create<TFL::MulOp>(
      divOp.getLoc(), quantOp.getResult().getType(),
      lhsDequantOp.getInput(), lookupOp.getResult(), divOp.getFusedActivationFunction());

    rewriter.replaceOp(quantOp, mulOp.getResult());
    rewriter.eraseOp(divOp);
    rewriter.eraseOp(lhsDequantOp);
    rewriter.eraseOp(rhsDequantOp);

    return failure();
  }
};

// Fold Dequant -> Quant into Quant
struct FoldDequantQuantPairPattern
    : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequantOp,
                                PatternRewriter &rewriter) const override {
    // Ensure the DequantizeOp has a single use
    if (!dequantOp->hasOneUse())
      return failure();
    // Ensure after DequantizeOp is QuantizeOp
    TFL::QuantizeOp quantOp = dyn_cast_or_null<TFL::QuantizeOp>(*dequantOp->getUsers().begin());
    if (!quantOp)
      return failure();

    // Get quantOp output type
    RankedTensorType outputType =
        quantOp.getResult().getType().dyn_cast<RankedTensorType>();

    auto newQuantOp = rewriter.create<TFL::QuantizeOp>(
        dequantOp.getLoc(), outputType, dequantOp.getInput(), 
        TypeAttr::get(outputType));

    rewriter.replaceOp(quantOp, newQuantOp.getResult());

    // Remove the dequant ops
    rewriter.eraseOp(dequantOp);

    return success();
  }
};

void OptimizeUnaryFloatOp::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);

  patterns.insert<MoveDequantForwardAndReplaceSqrtPattern>(ctx);
  patterns.insert<MoveDequantForwardOverUnaryOpPattern>(ctx);
  patterns.insert<MoveDequantForwardOverSameInputOpPattern>(ctx);
  patterns.insert<FoldDequantMulQuantOpPattern>(ctx);
  patterns.insert<FoldDequantDivQuantOpPattern>(ctx);
  patterns.insert<FoldDequantQuantPairPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OptimizeUnaryFloatOp pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeUnaryFloatOpPass() {
  return std::make_unique<OptimizeUnaryFloatOp>();
}

static PassRegistration<OptimizeUnaryFloatOp> pass;

} // namespace mlir::xcore
