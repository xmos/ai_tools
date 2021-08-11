// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "flatbuffers/flexbuffers.h"
#include "lib_nn/api/Conv2d.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

std::vector<uint8_t> FullyConnectedOp::buildCustomOptions() { return {}; }
std::vector<uint8_t> Lookup8Op::buildCustomOptions() { return {}; }

std::vector<uint8_t> PadOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("pad_value", (int32_t)pad_value()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Conv2DV2Op::buildCustomOptions() {
  int threadCount = (int)thread_count();

  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto threadsVec = fbb.StartVector("threads");
  for (int i = 0; i < threadCount; ++i) {
    auto vec = fbb.StartVector();
    fbb.Int((int32_t)scratch_bytes()
                .cast<ArrayAttr>()[i]
                .cast<IntegerAttr>()
                .getInt());
    fbb.Int((int32_t)(symbolizeConv2DType(conv2d_kernel_type()
                                              .cast<ArrayAttr>()[i]
                                              .cast<StringAttr>()
                                              .getValue()
                                              .str())
                          .getValue()));
    fbb.String(abstract_kernel_params()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str());
    fbb.String(memcpy_fn_params()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str());
    fbb.String(aggregate_fn_params()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str());
    fbb.String(output_transform_fn_params()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str());
    fbb.EndVector(vec, false, false);
  }
  fbb.EndVector(threadsVec, false, false);

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

namespace {
/// This pass translates XCore ops to TFLite custom ops.
struct TranslateToCustomOp
    : public PassWrapper<TranslateToCustomOp, FunctionPass> {
  void runOnFunction() override;
};

template <typename XCoreOp>
struct RewriteToCustomOp : public OpRewritePattern<XCoreOp> {
  using OpRewritePattern<XCoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XCoreOp xc_op,
                                PatternRewriter &rewriter) const override {
    auto options = xc_op.buildCustomOptions();
    auto *op = xc_op.getOperation();
    auto type = RankedTensorType::get({static_cast<int64_t>(options.size())},
                                      rewriter.getIntegerType(8));
    std::string options_bytes(options.begin(), options.end());
    auto attr = OpaqueElementsAttr::get(op->getDialect(), type, options_bytes);

    rewriter.replaceOpWithNewOp<TFL::CustomOp>(
        op, op->getResultTypes(), op->getOperands(),
        "XC_" + std::string(XCoreOp::getOperationName().drop_front(3)), attr);
    return success();
  }
};

void TranslateToCustomOp::runOnFunction() {
  auto *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  auto func = getFunction();

  patterns.insert<RewriteToCustomOp<FullyConnectedOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Lookup8Op>>(ctx);
  patterns.insert<RewriteToCustomOp<PadOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Conv2DV2Op>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the TranslateToCustomOp pass.
std::unique_ptr<OperationPass<FuncOp>> createTranslateToCustomOpPass() {
  return std::make_unique<TranslateToCustomOp>();
}

static PassRegistration<TranslateToCustomOp>
    pass("xcore-translate-to-customop",
         "Translate to custom ops in TensorFlow Lite dialect");

} // namespace xcore
} // namespace mlir
