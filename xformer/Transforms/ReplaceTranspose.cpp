// Replace TFL Transpose with Transpose for XCore.
#include "IR/XCoreOps.h"
#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {

// Replace TFL Transpose with xcore Transpose for XCore.
struct ReplaceTranspose
    : public PassWrapper<ReplaceTranspose, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceTranspose)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect, xcore::XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-transpose"; }
  StringRef getDescription() const final {
    return "Replace TFL Transpose with xcore Transpose for XCore.";
  }
  void runOnOperation() override;
};

int twoConsecutive(const SmallVector<int64_t, 4> &perm) {
  for (int i = 0; i < (int)perm.size() - 1; ++i) {
    if (perm[i] + 1 == perm[i + 1]) {
      return i; // Return the index i, not perm[i]
    }
  }
  return -1;
}

struct ReplaceTransposePattern : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    auto inputType =
        transposeOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    Value permValue = transposeOp.getPerm();
    DenseIntElementsAttr permAttr;
    if (!matchPattern(permValue, m_Constant(&permAttr))) {
      return failure();
    }

    SmallVector<int64_t, 4> perm;
    for (auto val : permAttr.getValues<APInt>()) {
      perm.push_back(val.getSExtValue());
    }

    SmallVector<int64_t, 4> reducedShape;
    SmallVector<int64_t, 4> reducedPerm;
    {
      SmallVector<int, 4> oldToNew;
      oldToNew.reserve(rank);
      for (int i = 0; i < rank; ++i) {
        if (inputShape[i] != 1) {
          oldToNew.push_back((int)reducedShape.size());
          reducedShape.push_back(inputShape[i]);
        } else {
          oldToNew.push_back(-1);
        }
      }

      for (auto p : perm) {
        if (oldToNew[p] != -1) {
          reducedPerm.push_back(oldToNew[p]);
        }
      }

      if (reducedShape.empty()) {
        reducedShape.push_back(1);
        reducedPerm.push_back(0);
      }
    }

    const size_t dtype_size = utils::getTypeSize(inputType.getElementType());
    if (dtype_size != 1) {
      reducedShape.push_back((int64_t)dtype_size);
      reducedPerm.push_back((int64_t)reducedShape.size() - 1);
    }
    auto mergeConsecutiveDims = [&](SmallVector<int64_t, 4> &shape,
                                    SmallVector<int64_t, 4> &perm) {
      while (true) {
        int i = twoConsecutive(perm);
        if (i == -1)
          break;
        int64_t p1 = perm[i];
        int64_t p2 = perm[i + 1];

        shape[p1] *= shape[p2];
        shape.erase(shape.begin() + p2);
        perm.erase(perm.begin() + i + 1);
        for (int j = 0; j < (int)perm.size(); ++j) {
          if (perm[j] > p2) {
            perm[j] -= 1;
          }
        }
      }
    };

    mergeConsecutiveDims(reducedShape, reducedPerm);

    if (reducedShape.size() > 4) {
      return failure();
    }

    // If size of reducedShape < 4, pad with 1's at the beginning
    // After padding shapes, adjust perm so that it matches
    int dimCount = (int)reducedShape.size();
    int pad = 4 - dimCount;
    if (pad > 0) {
      // Insert 1's at the start of shape
      SmallVector<int64_t, 4> paddedShape;
      SmallVector<int64_t, 4> paddedPerm;
      paddedShape.resize(4, 1); // fill with ones
      for (int i = 0; i < dimCount; ++i) {
        paddedShape[pad + i] = reducedShape[i];
      }

      for (int i = 0; i < pad; ++i) {
        paddedPerm.push_back(i);
      }
      for (auto p : reducedPerm) {
        paddedPerm.push_back(p + pad);
      }

      reducedShape = paddedShape;
      reducedPerm = paddedPerm;
    }

    const int RANK = 4;
    SmallVector<int32_t, 4> offsets(RANK);
    for (int i = 0; i < RANK; ++i) {
      int32_t prod = 1;
      for (int j = i + 1; j < RANK; ++j) {
        prod *= (int32_t)reducedShape[j];
      }
      offsets[i] = prod;
    }

    SmallVector<int32_t, 4> permutedOffsets(RANK);
    for (int i = 0; i < RANK; ++i) {
      permutedOffsets[i] = offsets[reducedPerm[i]];
    }

    // t_shape = tuple(SHAPE[p] for p in PERM)
    SmallVector<int32_t, 4> tShape(RANK);
    for (int i = 0; i < RANK; ++i) {
      tShape[i] = (int32_t)reducedShape[reducedPerm[i]];
    }

    auto outputType = transposeOp.getOutput().getType();
    auto newTransposeOp = rewriter.create<TransposeOp>(
        transposeOp.getLoc(), outputType, transposeOp.getInput(),
        rewriter.getI32ArrayAttr(permutedOffsets),
        rewriter.getI32ArrayAttr(tShape));

    rewriter.replaceOp(transposeOp, newTransposeOp.getResult());

    return success();
  }
};

void ReplaceTranspose::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceTransposePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the ReplaceTranspose pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceTransposePass() {
  return std::make_unique<ReplaceTranspose>();
}

static PassRegistration<ReplaceTranspose> pass;

} // namespace mlir::xcore
