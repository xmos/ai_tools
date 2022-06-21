// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace  Concat
struct ReplaceConcat
    : public PassWrapper<ReplaceConcat, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
   StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concat.";
  }
  void runOnFunction() override;
};

struct ReplaceConcatPattern
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {
    

    auto numInputs = concatOp.values().size();

    auto input0 = concatOp.values().operator[](0);
    auto input1 = concatOp.values().operator[](1);

    auto inputType0 = input0.getType().dyn_cast<RankedTensorType>();
    auto inputType1 = input1.getType().dyn_cast<RankedTensorType>();
    
    auto inputHeight0 = inputType0.getDimSize(1);
    auto inputWidth0 = inputType0.getDimSize(2);
    auto inputDepth0 = inputType0.getDimSize(3);

    auto inputHeight1 = inputType1.getDimSize(1);
    auto inputWidth1 = inputType1.getDimSize(2);
    auto inputDepth1 = inputType1.getDimSize(3);

    auto image_geom0 = nn::ImageGeometry(inputHeight0, inputWidth0,
                                        static_cast<int>(inputDepth0));

    auto window_geom0 =
        nn::WindowGeometry({static_cast<int>(inputHeight0), static_cast<int>(inputWidth0), static_cast<int>(inputDepth0)},
                           {0, 0}, {1, 1, 1}, {1, 1});

    auto image_geom1 = nn::ImageGeometry(inputHeight1, inputWidth1,
                                        static_cast<int>(inputDepth1));

    auto window_geom1 =
        nn::WindowGeometry({static_cast<int>(inputHeight1), static_cast<int>(inputWidth1), static_cast<int>(inputDepth1)},
                           {0, 0}, {1, 1, 1}, {1, 1});
    
    auto image_geom = nn::ImageGeometry(inputHeight1, inputWidth1,
                                        static_cast<int>(inputDepth1));

    auto window_geom =
        nn::WindowGeometry({static_cast<int>(inputHeight1), static_cast<int>(inputWidth1), static_cast<int>(inputDepth1)},
                           {0, 0}, {1, 1, 1}, {1, 1});

    nn::ImToColValid::Params imToColParams0(image_geom0, window_geom0,static_cast<int>(inputDepth0));
    std::string mfStr0 = imToColParams0.serialise<nn::ImToColValid::Params>();

    nn::ImToColValid::Params imToColParams1(image_geom1, window_geom1,static_cast<int>(inputDepth1));
    std::string mfStr1 = imToColParams1.serialise<nn::ImToColValid::Params>();

    //needs updating
    nn::ImToColValid::Params imToColParams(image_geom, window_geom,static_cast<int>(inputDepth1));
    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
    
  

  auto output = concatOp.output();
  auto output_type = output.getType().dyn_cast_or_null<ShapedType>();
  if (!output_type) return failure();
  auto output_quantized_type =
      quant::QuantizedType::getQuantizedElementType(output_type);

 

  // auto new_output_type =
  //     RankedTensorType::get(output_type.getShape(), input_quantized_type);
  


    auto outputType =
        concatOp.output().getType().dyn_cast<RankedTensorType>();

    // Create the tensor
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto outputDepth = outputType.getDimSize(3);

    auto outputSize = outputHeight * outputWidth * outputDepth;
    std::vector<int8_t> dummy(outputSize, 0);

    ShapedType concatTensorType = RankedTensorType::get(
        outputSize, rewriter.getI8Type());
        // outputSize,  output_quantized_type);
    auto concatTensorAttr = DenseElementsAttr::get<int8_t>(concatTensorType, dummy);
    auto concatTensorOp =
      rewriter.create<ConstantOp>(concatOp.getLoc(), concatTensorAttr);
    
    int32_t offset0 = 0;
    int32_t offset1 = inputWidth0;

    auto copyIntoTensorOp0 = rewriter.create<CopyIntoTensorOp>(
        concatOp.getLoc(), concatOp.getType(),
        input0, concatTensorOp, offset0, rewriter.getStringAttr(mfStr0));

    auto copyIntoTensorOp1  = rewriter.create<CopyIntoTensorOp>(
        concatOp.getLoc(), concatOp.getType(),
         input1, concatTensorOp, offset1, rewriter.getStringAttr(mfStr1));
    
    auto connectorOp0  = rewriter.create<ConnectorOp>(
        concatOp.getLoc(), concatOp.getType(),
         copyIntoTensorOp0, copyIntoTensorOp1 );

    auto passThruOp  = rewriter.create<PassThruOp>(
        concatOp.getLoc(), concatOp.getType(),
         concatTensorOp, connectorOp0, rewriter.getStringAttr(mfStr));

    rewriter.replaceOp(concatOp, passThruOp.output());

    return success();
  }
};

void ReplaceConcat::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat> pass;

} // namespace xcore
} // namespace mlir
