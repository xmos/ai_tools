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
// Apply generated OpSplit patterns.
struct ApplyOpSplitPatterns : public PassWrapper<ApplyOpSplitPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
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

IntegerAttr getI32IntegerAttrZero(PatternRewriter &rewriter) { 
  int zeroValue = 0;
  return rewriter.getI32IntegerAttr(zeroValue);
} 

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

#include "Transforms/GeneratedOpSplitPatterns.inc"

void ApplyOpSplitPatterns::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyOpSplitPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyOpSplitPatternsPass() {
  return std::make_unique<ApplyOpSplitPatterns>();
}

static PassRegistration<ApplyOpSplitPatterns>
    pass("xcore-apply-opslitpatterns", "Apply generated OpSplit optimization patterns.");

} // namespace xcore
} // namespace mlir
