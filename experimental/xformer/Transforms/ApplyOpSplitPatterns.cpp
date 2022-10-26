// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include <numeric>
#include "lib_nn/api/MemCpyFn.hpp"

namespace mlir {
namespace xcore {

namespace {
struct InsertStridedSlicePatterns 
    : public PassWrapper<InsertStridedSlicePatterns, 
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertStridedSlicePatterns)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnOperation() override;
};

bool HasNoFollowingStridedSlice(Value outputVal) {
  if (outputVal.hasOneUse()) {
    if (llvm::isa<TFL::CustomOp>(*outputVal.getUsers().begin())) {
      auto op = dyn_cast<TFL::CustomOp>(*outputVal.getUsers().begin());
      if (op.custom_code().startswith("XC_Strided_Slice"))
        return false;
    }
  }
  return true;
}

bool HasFollowingReshape(Value outputVal) {
  if (outputVal.hasOneUse()) {
    if (llvm::isa<TFL::ReshapeOp>(*outputVal.getUsers().begin())) {
        return true;
    }
  }
  return false;
}

bool HasNoFollowingReshape(Value outputVal) {
  if (outputVal.hasOneUse()) {
    if (llvm::isa<TFL::ReshapeOp>(*outputVal.getUsers().begin())) {
        return false;
    }
  }
  return true;
}

IntegerAttr getI32IntegerAttrZero(PatternRewriter &rewriter) { 
  int zeroValue = 0;
  return rewriter.getI32IntegerAttr(zeroValue);
} 

// IntegerAttr getQuantization(PatternRewriter &rewriter, Value outputVal) { 
//   return  quant::CastQuantizedTypeAttrFromExpressedType(
//           rewriter, quantizeOp.qtypeAttr(),
//           quant::QuantizedType::castToExpressedType(
//               outputVal.getType()),

//           -1);;
// } 

// mlir::OperandRange getValues(mlir::OperandRange values) { 
//   return values;
// } 

// mlir::OperandRange getValues(mlir::OperandRange values) { 
//   return values;
// } 

StringAttr getMemcpyFnParam(PatternRewriter &rewriter, Value outputVal) {
  // Extract args from the op
  auto outputType =
      outputVal.getType().dyn_cast<RankedTensorType>();
  auto outputHeight = outputType.getDimSize(1);
  auto outputWidth = outputType.getDimSize(2);
  auto outputDepth = outputType.getDimSize(3);

  auto inputHeight = outputHeight;
  auto inputWidth = outputWidth;
  auto inputDepth = outputDepth;
  auto beginY = 0;
  auto beginX = 0;
  auto endY = outputHeight;
  auto endX = outputWidth;
  auto strideY = 1;
  auto strideX = 1;

  auto image_geom = nn::ImageGeometry(inputHeight, inputWidth,
                                        static_cast<int>(inputDepth));

  int xDiff = endX - beginX;
  int yDiff = endY - beginY;
  auto window_geom =
      nn::WindowGeometry({yDiff, xDiff, static_cast<int>(inputDepth)},
                          {beginY, beginX}, {1, 1, 1}, {strideY, strideX});

  nn::ImToColValid::Params imToColParams(image_geom, window_geom,
                                          static_cast<int>(inputDepth));

  std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();

  return rewriter.getStringAttr(mfStr);
}

static Value insertStridedSlice(PatternRewriter &rewriter, Operation *op,
                                Value conv_out  ,
                                Value input  ,
  Value filter  ,
  Value bias  ,
    IntegerAttr dilation_h_factor  ,
    IntegerAttr dilation_w_factor  ,
   StringAttr fused_activation_function  ,
   StringAttr padding  ,
    IntegerAttr stride_h  ,
    IntegerAttr stride_w ) {

  //  auto new_conv_out = rewriter.create<mlir::TFL::Conv2DOp>(
  //     op->getLoc(), conv_out.getType(),  input  ,
  //   filter  ,
  //   bias  ,
  //     dilation_h_factor  ,
  //     dilation_w_factor  ,
  //    fused_activation_function  ,
  //    padding  ,
  //     stride_h  ,
  //     stride_w );

      int beginX = 0;
      int beginY = 0;

      // Extract args from the op
  auto outputType =
      conv_out.getType().dyn_cast<RankedTensorType>();
  auto outputHeight = outputType.getDimSize(1);
  auto outputWidth = outputType.getDimSize(2);
  auto outputDepth = outputType.getDimSize(3);

  auto inputHeight = outputHeight;
  auto inputWidth = outputWidth;
  auto inputDepth = outputDepth;
  // auto beginY = 0;
  // auto beginX = 0;
  auto endY = outputHeight;
  auto endX = outputWidth;
  auto strideY = 1;
  auto strideX = 1;

  auto image_geom = nn::ImageGeometry(inputHeight, inputWidth,
                                        static_cast<int>(inputDepth));

  int xDiff = endX - beginX;
  int yDiff = endY - beginY;
  auto window_geom =
      nn::WindowGeometry({yDiff, xDiff, static_cast<int>(inputDepth)},
                          {beginY, beginX}, {1, 1, 1}, {strideY, strideX});

  nn::ImToColValid::Params imToColParams(image_geom, window_geom,
                                          static_cast<int>(inputDepth));

  std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();

  return rewriter.create<mlir::xcore::StridedSliceOp>(
     op->getLoc(), conv_out.getType(),
        conv_out, rewriter.getI32IntegerAttr(beginX),
        rewriter.getI32IntegerAttr(beginY), rewriter.getStringAttr(mfStr));
}

#include "Transforms/GeneratedInsertStridedSlicePatterns.inc"

void InsertStridedSlicePatterns::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  func::FuncOp func = getOperation();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

  


} // namespace

// Creates an instance of the InsertStridedSlicePatterns pass.
std::unique_ptr<OperationPass<func::FuncOp>> createInsertStridedSlicePatternsPass() {
  return std::make_unique<InsertStridedSlicePatterns>();
}

static PassRegistration<InsertStridedSlicePatterns>
    pass();
    // pass("xcore-apply-opslitpatterns", "Apply generated OpSplit optimization patterns.");

} // namespace xcore
} // namespace mlir
