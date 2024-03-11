// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "flatbuffers/flexbuffers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

std::vector<uint8_t> Bsign8Op::buildCustomOptions() { return {}; }

std::vector<uint8_t> UnaryI16Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("type", (int32_t)getOpType()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> BinaryI16Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("type", (int32_t)getOpType()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Beta_ActivationF32Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("type", (int32_t)getType()); });
  fbb.Finish();
  return fbb.GetBuffer();
}
std::vector<uint8_t> Beta_ConcatF32Op::buildCustomOptions() { return {}; }
std::vector<uint8_t> Beta_ConvF32Op::buildCustomOptions() { return {}; }
std::vector<uint8_t> Beta_TransposeConvF32Op::buildCustomOptions() {
  return {};
}
std::vector<uint8_t> Beta_FcF32Op::buildCustomOptions() { return {}; }
std::vector<uint8_t> LookupOp::buildCustomOptions() { return {}; }
std::vector<uint8_t> SoftmaxOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> AddOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("m1", (int32_t)getMultiplier1());
    fbb.Int("m2", (int32_t)getMultiplier2());
    fbb.Int("bias", (int32_t)getBias());
    fbb.Int("shift", (int32_t)getShift());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> MulOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.String("mp", getMulParams().str()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> SliceOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto beginVec = fbb.StartVector("b");
  for (auto b : getBegin()) {
    fbb.Int((int32_t)b.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(beginVec, false, false);
  auto endVec = fbb.StartVector("e");
  for (auto e : getEnd()) {
    fbb.Int((int32_t)e.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(endVec, false, false);
  auto inOffsetVec = fbb.StartVector("i");
  for (auto i : getInputOffset()) {
    fbb.Int((int32_t)i.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(inOffsetVec, false, false);
  auto outOffsetVec = fbb.StartVector("o");
  for (auto o : getOutputOffset()) {
    fbb.Int((int32_t)o.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(outOffsetVec, false, false);

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> PadOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.String("pp", getPaddingPlan().str());
    fbb.Int("pv", (int32_t)getPadValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> PadOpV2::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto beginVec = fbb.StartVector("b");
  for (auto b : getBegin()) {
    fbb.Int((int32_t)b.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(beginVec, false, false);
  auto endVec = fbb.StartVector("e");
  for (auto e : getEnd()) {
    fbb.Int((int32_t)e.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(endVec, false, false);
  auto inOffsetVec = fbb.StartVector("i");
  for (auto i : getInputOffset()) {
    fbb.Int((int32_t)i.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(inOffsetVec, false, false);
  auto outOffsetVec = fbb.StartVector("o");
  for (auto o : getOutputOffset()) {
    fbb.Int((int32_t)o.cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(outOffsetVec, false, false);

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> LoadFlashOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  fbb.Int("addr", (int32_t)getAddress());
  auto sizesVec = fbb.StartVector("sizes");
  for (int i = 0; i < getSizes().cast<ArrayAttr>().size(); ++i) {
    fbb.Int(getSizes().cast<ArrayAttr>()[i].cast<IntegerAttr>().getInt());
  }
  fbb.EndVector(sizesVec, false, false);
  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Pad3To4Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("pv", (int32_t)getPadValue()); });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> Conv2DV2Op::buildCustomOptions() {
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  // TODO: Create a flatbuffer schema for xc ops.
  // The flexbuffer data for xc conv2d has been carefully arranged so
  // that each param is aligned to four bytes.
  // DO NOT CHANGE THE NAMES OR ORDER OF PARAMS HERE.
  // This is so that we can directly access them without creating
  // persistent buffers.
  // The alignment is why we are adding a dummy "00" to the end of
  // abstract kernel params. This is necessary when we have multiple
  // threads.
  fbb.String("mp", getMemcpyFnParam().str());
  fbb.String("a", getAggregateFnParam().str());
  fbb.String("o", getOutputTransformFnParam().str());
  int threadCount = (int)getThreadCount();
  auto akpVec = fbb.StartVector("p");
  for (int i = 0; i < threadCount; ++i) {
    fbb.String(getAbstractKernelParams()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str() +
               "00");
  }
  fbb.EndVector(akpVec, false, false);
  fbb.Int("s", (int32_t)getScratchBytes());
  fbb.Int("k", (int32_t)(symbolizeConv2DType(getConv2dKernelType()).value()));
  fbb.Int("t", (int32_t)(symbolizeOtType(getOutputTransformType()).value()));

  fbb.EndMap(rootMap);
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> MaxPool2DOp::buildCustomOptions() {
  // TODO: Is the alignement messed up?
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  fbb.String("mp", getMemcpyFnParam().str());
  fbb.String("a", getAggregateFnParam().str());
  fbb.String("o", getOutputTransformFnParam().str());
  int threadCount = (int)getThreadCount();
  auto akpVec = fbb.StartVector("p");
  for (int i = 0; i < threadCount; ++i) {
    fbb.String(getAbstractKernelParams()
                   .cast<ArrayAttr>()[i]
                   .cast<StringAttr>()
                   .getValue()
                   .str() +
               "00");
  }
  fbb.EndVector(akpVec, false, false);
  fbb.Int("s", (int32_t)getScratchBytes());

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
    auto attr = TFL::ConstBytesAttr::get(op->getContext(), options_bytes);

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
  patterns.insert<RewriteToCustomOp<MaxPool2DOp>>(ctx);
  patterns.insert<RewriteToCustomOp<LoadFlashOp>>(ctx);
  patterns.insert<RewriteToCustomOp<LookupOp>>(ctx);
  patterns.insert<RewriteToCustomOp<SoftmaxOp>>(ctx);
  patterns.insert<RewriteToCustomOp<MulOp>>(ctx);
  patterns.insert<RewriteToCustomOp<Pad3To4Op>>(ctx);
  patterns.insert<RewriteToCustomOp<SliceOp>>(ctx);
  patterns.insert<RewriteToCustomOp<PadOp>>(ctx);
  patterns.insert<RewriteToCustomOp<PadOpV2>>(ctx);
  patterns.insert<RewriteToCustomOp<Beta_ActivationF32Op>>(ctx);
  patterns.insert<RewriteToCustomOp<Beta_ConcatF32Op>>(ctx);
  patterns.insert<RewriteToCustomOp<Beta_ConvF32Op>>(ctx);
  patterns.insert<RewriteToCustomOp<Beta_TransposeConvF32Op>>(ctx);
  patterns.insert<RewriteToCustomOp<Beta_FcF32Op>>(ctx);
  patterns.insert<RewriteToCustomOp<UnaryI16Op>>(ctx);
  patterns.insert<RewriteToCustomOp<BinaryI16Op>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the TranslateToCustomOp pass.
std::unique_ptr<OperationPass<func::FuncOp>> createTranslateToCustomOpPass() {
  return std::make_unique<TranslateToCustomOp>();
}

static PassRegistration<TranslateToCustomOp> pass;

} // namespace mlir::xcore
