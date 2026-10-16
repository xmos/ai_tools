#include "IR/XCoreOps.h"
#include "Utils/Util.h"
#include "lib_nn/api/AggregateFn.hpp"
#include "lib_nn/api/MemCpyFn.hpp"
#include "lib_nn/api/OutputTransformFn.hpp"
extern "C" {
#include "lib_nn/api/nn_layers.h"
}
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir::xcore {

namespace {
// Replace TFL Batch MatMul with Batch MatMul for XCore.
struct ReplaceBatchMatMul
    : public PassWrapper<ReplaceBatchMatMul, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceBatchMatMul)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-batch-matmul"; }
  StringRef getDescription() const final {
    return "Replace TFL Batch MatMul with Batch MatMul for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceBatchMatMulPattern : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::BatchMatMulOp batchMatMulOp,
                                PatternRewriter &rewriter) const override {

    auto X = batchMatMulOp.getX();
    auto Y = batchMatMulOp.getY();
    auto output = batchMatMulOp.getOutput();
    bool memcpyOnX = true;
    bool reorderOnY = true;

    auto XType = X.getType().cast<ShapedType>();
    auto YType = Y.getType().cast<ShapedType>();
    auto outputType = output.getType().cast<ShapedType>();
    
    // Check if input and output are int8.
    bool isInt8 = utils::isNBitSignedQType<8>(XType.getElementType()) &&
                  utils::isNBitSignedQType<8>(YType.getElementType()) &&
                  utils::isNBitSignedQType<8>(outputType.getElementType());

    if (!isInt8) {
      return failure();
    }

    auto XShape = XType.getShape();
    if (XShape.size() != 3) {
      return mlir::failure(); // We don't support batch matmul with rank != 3
    }

    auto YShape = YType.getShape();
    if (YShape.size() != 3) {
      return mlir::failure(); // We don't support batch matmul with rank != 3
    }

    if (XShape[0] != YShape[0] || XShape[2] != YShape[1]) {
      return mlir::failure(); // X Y shape not match
    }

    int64_t B = XShape[0];
    int64_t N = XShape[1];
    int64_t M = YShape[1];
    int64_t K = YShape[2];

    auto XConstOp = dyn_cast_or_null<TFL::QConstOp>(X.getDefiningOp());
    auto YConstOp = dyn_cast_or_null<TFL::QConstOp>(Y.getDefiningOp());
    
    if (XConstOp) {
      llvm::outs() << "const op on x\n";
    }

    if (YConstOp) {
      llvm::outs() << "const op on y\n";
    }

    if (XConstOp && YConstOp) {
      // Shouldn't be both const
      return failure();
    }

    bool adjX = batchMatMulOp.getAdjX();
    bool adjY = batchMatMulOp.getAdjY();

    // Here we take the chance to reorder X or Y if they are const
    if (YConstOp) {
      auto valueAttr = YConstOp.getValue().cast<DenseElementsAttr>();
      std::vector<int8_t> dataVec = std::vector<int8_t>(
        valueAttr.getValues<int8_t>().begin(), valueAttr.getValues<int8_t>().end());
      llvm::outs() << "shape " << B << " " << M << " " << K << " total: " << dataVec.size() << "\n";
      llvm::outs() << "new shape " << B << " " << K << " " << M << "\n";

      if (adjY == false) {
        // Reorder to column-major (B, K, M), adjY == true means it's already column-major
        std::vector<int8_t> oldVec = dataVec;
        dataVec = std::vector<int8_t>(B*M*K);
        for (int64_t b = 0; b < B; ++b) {
          for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
              dataVec[b * (K * M) + k * M + m] = oldVec[b * (K * M) + m * K + k];
            }
          }
        }
      }

      // Create new constant attribute with shape [B, K, M]
      TypeAttr newTypeAttr = TypeAttr::get(
          RankedTensorType::get(
            {B, K, M}, YConstOp.getQtype().cast<RankedTensorType>().getElementType()));
      DenseElementsAttr newValueAttr = DenseElementsAttr::get<int8_t>(
          RankedTensorType::get(
            {B, K, M}, rewriter.getIntegerType(8)), dataVec);
      Y = rewriter.create<TFL::QConstOp>(
          batchMatMulOp.getLoc(), newTypeAttr, newValueAttr);
      reorderOnY = false;
    }

    auto xQType = utils::getQType(X);
    auto yQType = utils::getQType(Y);
    auto outputQType = utils::getQType(output);

    float xZeroPoint = static_cast<float>(xQType.getZeroPoint());
    float yZeroPoint = static_cast<float>(yQType.getZeroPoint());
    float outZeroPoint = static_cast<float>(outputQType.getZeroPoint());
    float xScale = static_cast<float>(xQType.getScale());
    float yScale = static_cast<float>(yQType.getScale());
    float outScale = static_cast<float>(outputQType.getScale());

    int32_t computeShape[4] = {B, N, M, K};  // batch, lhs row size, channel_size, rhs col size

    auto xcBatchMatMulOp = rewriter.create<BatchMatMulOp>(
        batchMatMulOp.getLoc(), batchMatMulOp.getType(), X, Y,
        rewriter.getI32ArrayAttr(computeShape),
        rewriter.getF32FloatAttr(xZeroPoint),
        rewriter.getF32FloatAttr(yZeroPoint),
        rewriter.getF32FloatAttr(outZeroPoint),
        rewriter.getF32FloatAttr(xScale*yScale/outScale)
      );
    rewriter.replaceOp(batchMatMulOp, xcBatchMatMulOp.getOutput());
    
    return success();
  }
};

void ReplaceBatchMatMul::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceBatchMatMulPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceBatchMatMul pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceBatchMatMulPass() {
  return std::make_unique<ReplaceBatchMatMul>();
}

static PassRegistration<ReplaceBatchMatMul> pass;

} // namespace mlir::xcore
