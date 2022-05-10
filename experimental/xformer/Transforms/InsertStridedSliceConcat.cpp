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
// Replace TFL StridedSlice with StridedSlice for XCore.
struct InsertStridedSliceConcat
    : public PassWrapper<InsertStridedSliceConcat, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct InsertStridedSliceConcatPattern
    : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv2DOp,
                                PatternRewriter &rewriter) const override {

    auto outputVal = conv2DOp.output();
    if (not outputVal.use_empty()) {
      if (llvm::isa<TFL::StridedSliceOp>(
                    *outputVal.getUsers().begin())) {
        return failure();
      }
    }


    // auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
    //     conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(),
    //     conv2DOp.filter(),
    //     conv2DOp.bias(),
    //     conv2DOp.dilation_h_factor(),
    //     conv2DOp.dilation_w_factor(),
    //     conv2DOp.fused_activation_function(),
    //     conv2DOp.padding(),
    //     conv2DOp.stride_h(),
    //     conv2DOp.stride_w()
    //     );

    // Extract args from the op
    auto outputType =
        outputVal.getType().dyn_cast<RankedTensorType>();
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto outputDepth = outputType.getDimSize(3);

    // std::vector<int32_t> stridedSlice_begin(4, 0);
    // // std::vector<int32_t> stridedSlice_begin = (0,0,0,0);
    // std::vector<int32_t> stridedSlice_end = (1,outputHeight, outputWidth, outputDepth);
    // std::vector<int32_t> stridedSlice_strides(4,1);
    // // std::vector<int32_t> stridedSlice_strides = (1,1,1,1);
    // IntegerAttr begin_mask = 0;
    // IntegerAttr end_mask = 0;
    // IntegerAttr ellipsis_mask = 0;
    // IntegerAttr new_axis_mask = 0;
    // IntegerAttr shrink_axis_mask = 0;

    // auto newStridedSliceOp = rewriter.create<TFL::StridedSliceOp>(
    //   conv2DOp.getLoc(), conv2DOp.getType(),conv2DOp.output(),
    //   stridedSlice_begin,
    //   stridedSlice_end,
    //   stridedSlice_strides,
    //   begin_mask,
    //   end_mask,
    //   ellipsis_mask,
    //   new_axis_mask,
    //   shrink_axis_mask 
      // );
    
    auto inputHeight = outputHeight;
    auto inputWidth = outputWidth;
    auto inputDepth = outputDepth;
    auto begin_y = 0;
    auto begin_x = 0;
    auto end_y = outputHeight;
    auto end_x = outputWidth;
    auto stride_y = 1;
    auto stride_x = 1;

    auto image_geom =
        nn::ImageGeometry(inputHeight, inputWidth, static_cast<int>(inputDepth));
    
    int x_diff = end_x - begin_x;
    int y_diff = end_y - begin_y;
    auto window_geom = nn::WindowGeometry(
        {y_diff,
         x_diff, static_cast<int>(inputDepth)},
        {begin_y, begin_x}, {1, 1, 1}, {stride_y, stride_x});

    nn::ImToColValid::Params imToColParams(image_geom, window_geom,static_cast<int>(inputDepth));

    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();

    llvm::SmallVector<std::string> strParams;

    strParams.push_back(mfStr);

    std::string memcpyFnParam;

    memcpyFnParam = strParams[0];

    rewriter.setInsertionPointAfter(conv2DOp);

    auto binaryObjectStridedSliceOp = rewriter.create<StridedSliceV3Op>(
        conv2DOp.getLoc(),conv2DOp.getType(),conv2DOp.output(),
        rewriter.getI32IntegerAttr(begin_x),
        rewriter.getI32IntegerAttr(begin_y),
        rewriter.getStringAttr(memcpyFnParam));

    // rewriter.insert(binaryObjectStridedSliceOp);

    return success();
  }
};

void InsertStridedSliceConcat::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InsertStridedSliceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the InsertStridedSliceConcat pass.
std::unique_ptr<OperationPass<FuncOp>> createInsertStridedSliceConcatPass() {
  return std::make_unique<InsertStridedSliceConcat>();
}

static PassRegistration<InsertStridedSliceConcat>
    pass("insert-stridedslice-concat",
         "InsertStridedSliceConcat.");

} // namespace xcore
} // namespace mlir
