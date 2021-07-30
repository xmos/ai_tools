// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/Conv2d.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Replace TFL Conv2D with XC Conv2DV2 ops.
struct ReplaceWithConv2DV2
    : public PassWrapper<ReplaceWithConv2DV2, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

struct ReplaceWithConv2DV2Pattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv2DOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8
    if (!(conv2DOp.input()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<quant::QuantizedType>() &&
          conv2DOp.input()
                  .getType()
                  .cast<ShapedType>()
                  .getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Check if not fitting the three kernel types, then return
    // Output channel must be a multiple of four
    if (!conv2DOp.output().getType().cast<ShapedType>().getDimSize(3) % 4 ==
        0) {
      return failure();
    }

    // Get thread count as command-line option
    // Currently thread count is one
    int threadCount = 1;

    // Multithread analysis to determine how to split up the data between
    // threads.
    // Also to determine which kernel type for each thread.
    // Might be better to do this as an analysis pass and access the analysis
    // results here

    for (int i = 0; i < threadCount; ++i) {
      // Retrieve some of the parameters from the op
      // This is to find the correct kernel type

      // Retrieve the rest of the parameters from the op
      // Call the function for that which returns a vector of four strings for
      // the four params

      // Find out the scratch size needed calling the lib_nn functions

      // have to accumulate the vector of strings to be written out later
    }

    // Create the Conv2DV2 Op with the params and kernel type

    void *w, *x, *y, *z;
    nn::Conv2dValidDirect k((nn::AbstractKernel::Params *)&w,
                            (nn::DerefInputFn *)&x, (nn::MatMulDirectFn *)&y,
                            (nn::OT_int8 *)&z);

    std::vector<uint8_t> dummy(10, 100);

    auto data = dummy;
    auto type = RankedTensorType::get({static_cast<int64_t>(data.size())},
                                      rewriter.getIntegerType(8));
    std::string options_bytes(data.begin(), data.end());
    auto attr = OpaqueElementsAttr::get(
        Identifier::get(XCoreDialect::getDialectNamespace(),
                        rewriter.getContext()),
        type, options_bytes);

    auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
        conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(), attr, attr,
        attr, attr,
        rewriter.getStringAttr(stringifyConv2DType(Conv2DType::ValidInDirect)));
    rewriter.replaceOp(conv2DOp, newConv2DV2Op.output());

    return success();
  }
};

void ReplaceWithConv2DV2::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceWithConv2DV2Pattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceWithConv2DV2 pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceWithConv2DV2Pass() {
  return std::make_unique<ReplaceWithConv2DV2>();
}

static PassRegistration<ReplaceWithConv2DV2>
    pass("xcore-replace-with-conv2dv2",
         "Replace TFL Conv2D with XC Conv2DV2 operations.");

} // namespace xcore
} // namespace mlir
