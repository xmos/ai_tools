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
std::vector<uint8_t> LookupOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> AddOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("m1", (int32_t)multiplier1());
    fbb.Int("m2", (int32_t)multiplier2());
    fbb.Int("bias", (int32_t)bias());
    fbb.Int("shift", (int32_t)shift());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

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
  auto rootMap = fbb.StartMap();
  fbb.Int("addr", (int32_t)address());
  auto sizesVec = fbb.StartVector("sizes");
  for (int i = 0; i < sizes().cast<ArrayAttr>().size(); ++i) {
    fbb.Int(sizes().cast<ArrayAttr>()[i].cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(sizesVec, false, false);
  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> PadOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.String("pp", padding_plan().str());
    fbb.Int("pv", (int32_t)pad_value());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Pad3To4Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("pv", (int32_t)pad_value()); });
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
  fbb.Int("ott",
          (int32_t)(symbolizeOtType(output_transform_type()).getValue()));
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
    : public PassWrapper<TranslateToCustomOp, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TranslateToCustomOp)

  StringRef getArgument() const final { return "xcore-translate-to-customop"; }
  StringRef getDescription() const final {
    return "Translate to custom ops in TensorFlow Lite dialect";
  }
  void runOnOperation() override;
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

void TranslateToCustomOp::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.insert<RewriteToCustomOp<AddOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Bsign8Op>>(ctx);
  patterns.insert<RewriteToCustomOp<Conv2DV2Op>>(ctx);
  patterns.insert<RewriteToCustomOp<LoadFlashOp>>(ctx);
  patterns.insert<RewriteToCustomOp<LookupOp>>(ctx);
  patterns.insert<RewriteToCustomOp<PadOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Pad3To4Op>>(ctx);
  patterns.insert<RewriteToCustomOp<StridedSliceOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the TranslateToCustomOp pass.
std::unique_ptr<OperationPass<func::FuncOp>> createTranslateToCustomOpPass() {
  return std::make_unique<TranslateToCustomOp>();
}

static PassRegistration<TranslateToCustomOp> pass;

} // namespace xcore
} // namespace mlir
