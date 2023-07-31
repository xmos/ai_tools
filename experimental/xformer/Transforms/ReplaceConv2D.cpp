// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/ConvPatterns.h"
#include "Transforms/Options.h"
#include "Utils/ThreadSupport.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

// XC Conv2D Base class implementation
// ConcreteType would be TFL Conv types or Larq BConv2D
// Replaces them with XC Conv2D
template <typename ConcreteType, typename ConvOpType, typename ArgsType>
LogicalResult
ReplaceWithXCConv2DBase<ConcreteType, ConvOpType, ArgsType>::matchAndRewrite(
    ConvOpType op, PatternRewriter &rewriter) const {
  auto conv2DOp = static_cast<ConvOpType>(op);
  auto builder = static_cast<const ConcreteType *>(this);

  // Check if the op has invalid types
  if (failed(builder->checkIfValid(conv2DOp))) {
    return failure();
  }

  // Extract common args from the op
  ArgsType args;
  args.convOp = conv2DOp.getOperation();
  auto inputType =
      conv2DOp.getInput().getType().template dyn_cast<RankedTensorType>();
  auto outputType =
      conv2DOp.getOutput().getType().template dyn_cast<RankedTensorType>();
  auto filterType =
      conv2DOp.getFilter().getType().template dyn_cast<RankedTensorType>();
  args.inputHeight = inputType.getDimSize(1);
  args.inputWidth = inputType.getDimSize(2);
  args.inputDepth = inputType.getDimSize(3);
  args.outputHeight = outputType.getDimSize(1);
  args.outputWidth = outputType.getDimSize(2);
  args.outputDepth = outputType.getDimSize(3);
  args.filterHeight = filterType.getDimSize(1);
  args.filterWidth = filterType.getDimSize(2);
  args.filterDepth = filterType.getDimSize(3);
  // Get op-type specific args
  if (failed(builder->getArgs(conv2DOp, args))) {
    return failure();
  }

  Conv2DType kernelType;
  if (failed(builder->getKernelType(args, kernelType))) {
    return failure();
  }

  OtType otType;

  llvm::SmallVector<std::string> abstractKernelParams;
  std::string memcpyFnParam, aggregateFnParam, outputTransformFnParam,
      kernelTypeEnumParam, otTypeEnumParam;

  int32_t scratchByteParam;

  std::vector<int8_t> weightsData;
  std::vector<int16_t> mulsBiasesOrThresholdsData;

  // Obtain thread count from command-line option
  const int threadCount = threadCountOption;
  llvm::SmallVector<std::string> strParams;
  int scratchBytes = 0;
  // Get image region splits for multiple threads
  args.imageRegionSplits = utils::getImageRegionThreadSplits(
      threadCount, args.Y.height, args.Y.width);

  // Obtain serialized params and calculated tensors from lib_nn for the
  // conv2d kernel type
  if (failed(builder->getSerializedParamsAndTensors(
          args, kernelType, otType, strParams, abstractKernelParams,
          weightsData, mulsBiasesOrThresholdsData, scratchBytes))) {
    return failure();
  }

  // The actual thread count might be less than the count specified on the
  // command-line
  // If the output height/width is smaller than the specified
  // thread count, only the required number of threads would be used
  // The number of abstract kernel params gives us the actual thread count
  int actualThreadCount = abstractKernelParams.size();

  // Prepare params to create Conv2DV2 Op
  kernelTypeEnumParam = stringifyConv2DType(kernelType).str();
  memcpyFnParam = strParams[0];
  aggregateFnParam = strParams[1];
  outputTransformFnParam = strParams[2];
  otTypeEnumParam = stringifyOtType(otType).str();
  scratchByteParam = scratchBytes;

  // Create a string array attr from a vector of strings
  auto getStringArrayAttr = [&](llvm::SmallVector<std::string> value) {
    auto attrs = llvm::to_vector<8>(
        llvm::map_range(value, [&](std::string v) -> Attribute {
          return rewriter.getStringAttr(v);
        }));
    return rewriter.getArrayAttr(attrs);
  };

  // Create the tensors for weights and multipliers_and_biases
  ShapedType weightsType = RankedTensorType::get(
      {static_cast<long long>(weightsData.size())}, rewriter.getIntegerType(8));
  auto weightsAttr = DenseElementsAttr::get<int8_t>(weightsType, weightsData);
  auto weightsConstantOp =
      rewriter.create<arith::ConstantOp>(conv2DOp.getLoc(), weightsAttr);

  ShapedType mulsBiasesOrThresholdsType = RankedTensorType::get(
      {static_cast<long long>(mulsBiasesOrThresholdsData.size())},
      rewriter.getIntegerType(16));
  auto mulsBiasesOrThresholdsAttr = DenseElementsAttr::get<int16_t>(
      mulsBiasesOrThresholdsType, mulsBiasesOrThresholdsData);
  auto mulsBiasesOrThresholdsConstantOp = rewriter.create<arith::ConstantOp>(
      conv2DOp.getLoc(), mulsBiasesOrThresholdsAttr);

  // Create the Conv2DV2 Op with the params and kernel type
  auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
      conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.getInput(),
      weightsConstantOp, mulsBiasesOrThresholdsConstantOp,
      rewriter.getStringAttr(kernelTypeEnumParam),
      rewriter.getStringAttr(memcpyFnParam),
      rewriter.getStringAttr(aggregateFnParam),
      rewriter.getStringAttr(outputTransformFnParam),
      rewriter.getStringAttr(otTypeEnumParam),
      rewriter.getI32IntegerAttr(scratchByteParam),
      rewriter.getI32IntegerAttr(actualThreadCount),
      getStringArrayAttr(abstractKernelParams));
  rewriter.replaceOp(conv2DOp, newConv2DV2Op.getOutput());

  return success();
}

namespace {
// Replace with XC Conv2D pass
struct ReplaceConv2D
    : public PassWrapper<ReplaceConv2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceConv2D)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<TFL::TensorFlowLiteDialect, XCoreDialect, lq::LarqDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-conv2d"; }
  StringRef getDescription() const final {
    return "Replace Conv2D with XC Conv2D pass";
  }
  void runOnOperation() override;
};

namespace convpatterns {
#include "Transforms/GeneratedConvPatterns.inc"
}

namespace convrevertpatterns {
#include "Transforms/GeneratedConvRevertPatterns.inc"
}

void ReplaceConv2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  // Apply patterns to lower TFL Conv to XC Fake Conv ops
  // This helps in pattern matching only types we support for xcore such as QI8
  // and for handling issues such as EXPLICIT padding which is not supported in
  // TFL Conv ops
  RewritePatternSet patterns1(ctx);
  convpatterns::populateWithGenerated(patterns1);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns1));

  // Replace with XC Conv2D op
  RewritePatternSet patterns2(ctx);
  patterns2.insert<ReplaceConv2DPattern, ReplaceDepthwiseConv2DPattern,
                   ReplaceBConv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2));

  // Revert remaining XC Fake Conv ops back to TFL Conv2D ops
  RewritePatternSet patterns3(ctx);
  convrevertpatterns::populateWithGenerated(patterns3);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns3));

  // We walk through all Conv2DV2 ops in the graph and find the maximum required
  // thread count
  // This is stored as an attribute in the module and pushed as metadata into
  // the flatbuffer
  unsigned int requiredThreadCount = 1;
  auto module = func->getParentOfType<ModuleOp>();
  if (auto attr = module->getAttr(xcRequiredThreadCountAttrName)) {
    requiredThreadCount = attr.cast<mlir::IntegerAttr>().getInt();
  }
  func.walk([&](Conv2DV2Op op) {
    requiredThreadCount = std::max(requiredThreadCount, op.getThreadCount());
  });
  // Store as an attribute in the module
  OpBuilder builder(func);
  module->setAttr(xcRequiredThreadCountAttrName,
                  builder.getI32IntegerAttr(requiredThreadCount));
}
} // namespace

// Creates an instance of the ReplaceConv2D pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConv2DPass() {
  return std::make_unique<ReplaceConv2D>();
}

static PassRegistration<ReplaceConv2D> pass;

} // namespace xcore
} // namespace mlir
