// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "flatbuffers/flexbuffers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

std::vector<uint8_t> Bsign8Op::buildCustomOptions() { return {}; }
std::vector<uint8_t> Lookup8Op::buildCustomOptions() { return {}; }

std::vector<uint8_t> StridedSliceOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();

  fbb.Int("begin_x", (int32_t)begin_x());
  fbb.Int("begin_y", (int32_t)begin_y());
  fbb.String("mp", memcpy_fn_param().str());

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> LoadFlashOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("addr", (int32_t)address());
    fbb.Int("size", (int32_t)size());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> PadOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("pad_value", (int32_t)pad_value()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Conv2DV2Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();

  fbb.Int("kt",
          (int32_t)(symbolizeConv2DType(conv2d_kernel_type()).getValue()));
  fbb.String("mp", memcpy_fn_param().str());
  fbb.String("aggp", aggregate_fn_param().str());
  fbb.String("otp", output_transform_fn_param().str());
  fbb.Int("scratch", (int32_t)scratch_bytes());

  int threadCount = (int)thread_count();
  auto akpVec = fbb.StartVector("akp");
  for (int i = 0; i < threadCount; ++i) {
    fbb.String(abstract_kernel_params()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str());
  }
  fbb.EndVector(akpVec, false, false);

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

namespace {
/// This pass translates XCore ops to TFLite custom ops.
struct TranslateToCustomOp
    : public PassWrapper<TranslateToCustomOp, FunctionPass> {
  StringRef getArgument() const final { return "xcore-translate-to-customop"; }
  StringRef getDescription() const final {
    return "Translate to custom ops in TensorFlow Lite dialect";
  }
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

  patterns.insert<RewriteToCustomOp<Lookup8Op>>(ctx);
  patterns.insert<RewriteToCustomOp<PadOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Conv2DV2Op>>(ctx);
  patterns.insert<RewriteToCustomOp<LoadFlashOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Bsign8Op>>(ctx);
  patterns.insert<RewriteToCustomOp<StridedSliceOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the TranslateToCustomOp pass.
std::unique_ptr<OperationPass<FuncOp>> createTranslateToCustomOpPass() {
  return std::make_unique<TranslateToCustomOp>();
}

static PassRegistration<TranslateToCustomOp> pass;

} // namespace xcore
} // namespace mlir
