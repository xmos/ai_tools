//  Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir::xcore {

namespace {
static constexpr char opSplitLabel[] = "opSplitLabel";
static constexpr char opSplitLabelNumSplits[] = "opSplitLabelNumSplits";

// OpSplit
struct OpSplit : public PassWrapper<OpSplit, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSplit)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-op-split"; }
  StringRef getDescription() const final { return "Op Split."; }
  void runOnOperation() override;
};

TFL::SliceOp createSliceOp(PatternRewriter &rewriter, Location loc, Value input,
                           ArrayRef<int32_t> begin, ArrayRef<int32_t> size,
                           Type outputElemType) {
  RankedTensorType outputType =
      RankedTensorType::get({1, size[1], size[2], size[3]}, outputElemType);
  auto beginConstantOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32TensorAttr(begin));
  auto sizeConstantOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32TensorAttr(size));
  auto sliceOp = rewriter.create<TFL::SliceOp>(loc, outputType, input,
                                               beginConstantOp, sizeConstantOp);
  sliceOp->setAttr(opSplitLabel, rewriter.getUnitAttr());
  return sliceOp;
}

template <typename TargetOp>
struct OpSplitHorizontalPattern : public OpRewritePattern<TargetOp> {
  using OpRewritePattern<TargetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TargetOp targetOp,
                                PatternRewriter &rewriter) const override {
    // Do not split ops already split
    if (!(targetOp->hasAttr(opSplitLabelNumSplits)))
      return failure();

    int numSplits = 0;
    auto attr = targetOp->getAttr(opSplitLabelNumSplits);
    numSplits = attr.template cast<mlir::IntegerAttr>().getInt();

    if (targetOp->hasAttr(opSplitLabel))
      return failure();

    // Output type must be QI8
    if (!utils::isNBitSignedQType<8>(
            utils::getValElementType(targetOp.getOutput())))
      return failure();

    // Data from target op needed later
    auto targetOutput = targetOp.getOutput();
    auto outputType =
        targetOutput.getType().template dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    // Clone the op as we want to replace it with the same op type but with
    // slices and concat inserted after it
    auto targetReplacement = llvm::cast<TargetOp>(rewriter.clone(*targetOp));

    // Apply label, so that the same op is not rewritten a second time.
    targetReplacement->setAttr(opSplitLabel, rewriter.getUnitAttr());

    // Will hold slice op to insert after target op
    SmallVector<Value> sliceOps;

    int32_t sliceHeight = outputHeight / numSplits;

    // The remainder will be distributed between the splits
    // to keep them about the same size
    int32_t sliceHeightRemainder = outputHeight % numSplits;

    // For loop uses end index of previous slice created
    // needs to initalized to zero for first slice
    int32_t prevEndIndex = 0;

    // Loops creates slices with correct params
    for (size_t i = 0; i < numSplits; i++) {
      // Distributes remainder between slices
      int32_t currentSliceHeight = sliceHeight;
      if (i < sliceHeightRemainder)
        currentSliceHeight++;

      int32_t begin[4] = {0, prevEndIndex, 0, 0};
      int32_t size[4] = {1, currentSliceHeight, outputWidth, outputDepth};
      auto sliceOp =
          createSliceOp(rewriter, targetReplacement.getLoc(), targetReplacement,
                        begin, size, outputType.getElementType());

      prevEndIndex += currentSliceHeight;

      // Store created slice op to use as input to concat
      sliceOps.push_back(sliceOp.getResult());
    }

    // Create concat op that concats on dim 1, height
    auto concatOp = rewriter.create<TFL::ConcatenationOp>(
        targetReplacement.getLoc(), targetOutput.getType(), sliceOps, 1,
        "NONE");

    // Replace target op with [cloned target op -> slices -> concat]
    rewriter.replaceOp(targetOp, concatOp.getOutput());

    return success();
  }
};

struct RaiseSliceHorizontalAddPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!((slice->hasAttr(opSplitLabel))))
      return failure();

    // If slice does not have a defining op, return failure
    if (!(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (!isa<TFL::AddOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    auto addOriginal = llvm::cast<TFL::AddOp>(slice.getInput().getDefiningOp());

    // Do not raise slice if op does not have op split label
    if (!(addOriginal->hasAttr(opSplitLabel)))
      return failure();

    auto sliceOutShape = utils::getValShape(slice.getOutput());

    // Create new slice for above add
    auto sliceLHS = llvm::cast<TFL::SliceOp>(rewriter.clone(*slice));
    sliceLHS.setOperand(0, addOriginal.getLhs());
    RankedTensorType sliceLHSType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(addOriginal.getLhs()));
    sliceLHS->getResult(0).setType(sliceLHSType);

    // Create new slice for above add
    auto sliceRHS = llvm::cast<TFL::SliceOp>(rewriter.clone(*slice));
    sliceRHS.setOperand(0, addOriginal.getRhs());
    RankedTensorType sliceRHSType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(addOriginal.getRhs()));
    sliceRHS->getResult(0).setType(sliceRHSType);

    auto addReplacement = llvm::cast<TFL::AddOp>(rewriter.clone(*addOriginal));
    RankedTensorType addReplacementType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(addOriginal.getOutput()));
    addReplacement->getResult(0).setType(addReplacementType);
    addReplacement.setOperand(0, sliceLHS);
    addReplacement.setOperand(1, sliceRHS);

    // replace slice with new slice -> new add
    rewriter.replaceOp(slice, addReplacement.getOutput());

    return success();
  }
};

template <typename ConvOp>
struct RaiseSliceHorizontalPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!(slice->hasAttr(opSplitLabel)))
      return failure();

    // If slice does not have a defining op, return failure
    if (!(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (!isa<ConvOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    // Get data from conv needed to raise slice
    auto convOriginal = llvm::cast<ConvOp>(slice.getInput().getDefiningOp());

    // Do not raise slice if op does not have op split label
    if (!(convOriginal->hasAttr(opSplitLabel)))
      return failure();

    auto convOriginalInput =
        convOriginal.getInput().getType().template cast<RankedTensorType>();
    auto inputHeight = convOriginalInput.getDimSize(1);
    auto inputWidth = convOriginalInput.getDimSize(2);
    auto inputChannels = convOriginalInput.getDimSize(3);

    auto convOriginalOutput =
        convOriginal.getOutput().getType().template cast<RankedTensorType>();
    auto outputHeight = convOriginalOutput.getDimSize(1);
    auto outputChannels = convOriginalOutput.getDimSize(3);

    auto filterType = convOriginal.getFilter()
                          .getType()
                          .template dyn_cast<RankedTensorType>();
    auto filterHeight = filterType.getDimSize(1);
    auto filterWidth = filterType.getDimSize(2);

    auto strideHeight = convOriginal.getStrideH();
    auto strideWidth = convOriginal.getStrideW();

    DenseElementsAttr attr;
    if (!matchPattern(slice.getBegin(), m_Constant(&attr))) {
      return failure();
    }
    auto beginIndex = attr.getValues<int32_t>()[1];
    if (!matchPattern(slice.getSize(), m_Constant(&attr))) {
      return failure();
    }
    auto sizeIndex = attr.getValues<int32_t>()[1];
    auto endIndex = beginIndex + sizeIndex;

    // Get original slice's output height
    auto sliceOutput = slice.getOutput().getType().cast<RankedTensorType>();
    auto sliceOutputHeight = sliceOutput.getDimSize(1);

    int32_t newEndIndex;
    int32_t newOutputHeight;
    int64_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0;

    int dilation_h_factor = 1;
    int dilation_w_factor = 1;
    int64_t newHeight, newWidth;
    tensorflow::Padding opPadding = convOriginal.getPadding() == "VALID"
                                        ? tensorflow::Padding::VALID
                                        : tensorflow::Padding::SAME;
    // Get pad values for conv op
    if (tensorflow::GetWindowedOutputSizeVerbose(
            inputHeight, filterHeight, dilation_h_factor, strideHeight,
            opPadding, &newHeight, &padTop,
            &padBottom) != tensorflow::OkStatus()) {
      return failure();
    }
    if (tensorflow::GetWindowedOutputSizeVerbose(
            inputWidth, filterWidth, dilation_w_factor, strideWidth, opPadding,
            &newWidth, &padLeft, &padRight) != tensorflow::OkStatus()) {
      return failure();
    }

    // Check if padding is same
    int32_t lostFraction = 0;
    if (convOriginal.getPadding() == "VALID") {
      if (endIndex == outputHeight) {
        newEndIndex = inputHeight;
        lostFraction =
            (inputHeight - filterHeight + strideHeight) % strideHeight;
      } else {
        newEndIndex = endIndex * strideHeight - strideHeight + filterHeight;
      }
    } else if (convOriginal.getPadding() == "SAME") {
      // Check if this is left most split
      if (beginIndex == 0) {
        newEndIndex =
            endIndex * strideHeight - strideHeight + filterHeight - padTop;
        padBottom = 0;

      } else if (endIndex == outputHeight) {
        newEndIndex = endIndex * strideHeight - strideHeight + filterHeight -
                      padTop - padBottom;
        padTop = 0;

      } else {
        newEndIndex =
            endIndex * strideHeight - strideHeight + filterHeight - padTop;
        padTop = 0;
        padBottom = 0;
      }
    }
    // Calculate new output height after raising slice above conv
    newOutputHeight = sliceOutputHeight * strideHeight - strideHeight +
                      filterHeight - padTop - padBottom + lostFraction;

    int32_t beginAttr[4] = {
        0, static_cast<int32_t>(newEndIndex - newOutputHeight), 0, 0};
    int32_t sizeAttr[4] = {1, static_cast<int32_t>(newOutputHeight),
                           static_cast<int32_t>(inputWidth),
                           static_cast<int32_t>(inputChannels)};
    auto sliceReplacement = createSliceOp(
        rewriter, slice.getLoc(), convOriginal.getInput(), beginAttr, sizeAttr,
        utils::getValElementType(convOriginal.getInput()));

    // Adjust shape for padding
    // For valid conv the shapes will not change since pad values are zero
    auto paddedHeight = newOutputHeight + padTop + padBottom;
    auto paddedWidth = inputWidth + padLeft + padRight;

    // If padding is same, create pad op to extract padding
    TFL::PadOp padOp;
    if (convOriginal.getPadding() == "SAME") {
      std::vector<int32_t> paddingValues{0,
                                         0,
                                         static_cast<int>(padTop),
                                         static_cast<int>(padBottom),
                                         static_cast<int>(padLeft),
                                         static_cast<int>(padRight),
                                         0,
                                         0};

      RankedTensorType paddingsType =
          RankedTensorType::get({4, 2}, rewriter.getI32Type());

      Value paddings = rewriter.create<TFL::ConstOp>(
          slice.getLoc(),
          DenseIntElementsAttr::get(paddingsType, paddingValues));

      auto paddedResultType = RankedTensorType::get(
          {1, paddedHeight, paddedWidth, inputChannels},
          utils::getValElementType(convOriginal.getInput()));

      padOp = rewriter.create<TFL::PadOp>(slice.getLoc(), paddedResultType,
                                          sliceReplacement, paddings);
    }

    auto convReplacement = llvm::cast<ConvOp>(rewriter.clone(*convOriginal));

    RankedTensorType newConvType = RankedTensorType::get(
        {1, (paddedHeight + strideHeight - filterHeight) / strideHeight,
         (paddedWidth + strideWidth - filterWidth) / strideWidth,
         outputChannels},
        utils::getValElementType(convOriginal.getOutput()));
    convReplacement->getResult(0).setType(newConvType);

    // if valid padding no need for pad op, connect to slice
    // else connect to pad op
    if (convOriginal.getPadding() == "VALID") {
      // Connect new conv's input to new slice
      convReplacement.setOperand(0, sliceReplacement);

    } else if (convOriginal.getPadding() == "SAME") {
      // Connect new conv's input to pad op
      convReplacement.setOperand(0, padOp);

      // Change padding on cloned conv to valid since
      // padding was extracted to pad op
      convReplacement->setAttr("padding", rewriter.getStringAttr("VALID"));
    }

    // replace slice with new slice -> new conv
    // or new slice -> pad -> new conv
    rewriter.replaceOp(slice, convReplacement.getOutput());

    return success();
  }
};

struct RaiseSliceHorizontalPadPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!(slice->hasAttr(opSplitLabel)))
      return failure();

    // If slice does not have a defining op, return failure
    if (!(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (!isa<TFL::PadOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    auto padOriginal = llvm::cast<TFL::PadOp>(slice.getInput().getDefiningOp());

    // Do not raise slice if op does not have op split label
    if (!(padOriginal->hasAttr(opSplitLabel)))
      return failure();

    // Get data from pad needed to raise slice
    auto padOriginalInput =
        padOriginal.getInput().getType().template cast<RankedTensorType>();
    auto inputHeight = padOriginalInput.getDimSize(1);
    auto inputWidth = padOriginalInput.getDimSize(2);
    auto inputChannels = padOriginalInput.getDimSize(3);

    auto padOriginalOutput =
        padOriginal.getOutput().getType().template cast<RankedTensorType>();
    auto outputHeight = padOriginalOutput.getDimSize(1);
    auto outputWidth = padOriginalOutput.getDimSize(2);
    auto outputChannels = padOriginalOutput.getDimSize(3);

    int64_t padVertical, padHorizontal;
    int64_t padTop, padBottom, padLeft, padRight;

    padVertical = outputHeight - inputHeight;
    padHorizontal = outputWidth - inputWidth;

    padTop = padVertical / 2;
    padBottom = padVertical - padTop;
    padLeft = padHorizontal / 2;
    padRight = padHorizontal - padLeft;

    // Get original slice's output height
    auto sliceOutput = slice.getOutput().getType().cast<RankedTensorType>();
    auto sliceOutputHeight = sliceOutput.getDimSize(1);
    auto sliceOutputWidth = sliceOutput.getDimSize(2);

    // get end index of slice
    DenseElementsAttr attr;
    if (!matchPattern(slice.getSize(), m_Constant(&attr))) {
      return failure();
    }
    auto sizeIndex = attr.getValues<int32_t>()[1];

    // Get begin index for slice
    if (!matchPattern(slice.getBegin(), m_Constant(&attr))) {
      return failure();
    }
    auto beginIndex = attr.getValues<int32_t>()[1];
    auto endIndex = beginIndex + sizeIndex;

    int32_t newEndIndex;
    int32_t newOutputHeight;

    if (beginIndex == 0) {
      newEndIndex = endIndex - padTop;
      padBottom = 0;

    } else if (endIndex == outputHeight) { // end
      newEndIndex = endIndex - padTop - padBottom;
      padTop = 0;

    } else {
      newEndIndex = endIndex - padTop;
      padTop = 0;
      padBottom = 0;
    }

    // Calculate new output height after raising slice above pad
    newOutputHeight = sliceOutputHeight - padTop - padBottom;

    int32_t beginAttr[4] = {
        0, static_cast<int32_t>(newEndIndex - newOutputHeight), 0, 0};
    int32_t sizeAttr[4] = {1, static_cast<int32_t>(newOutputHeight),
                           static_cast<int32_t>(inputWidth),
                           static_cast<int32_t>(inputChannels)};
    auto sliceReplacement = createSliceOp(
        rewriter, slice.getLoc(), padOriginal.getInput(), beginAttr, sizeAttr,
        utils::getValElementType(padOriginal.getInput()));

    // Adjust shape for padding
    auto paddedHeight = newOutputHeight + padTop + padBottom;
    auto paddedWidth = inputWidth + padLeft + padRight;

    DenseIntElementsAttr pad;
    if (!matchPattern(padOriginal.getPadding(), m_Constant(&pad))) {
      return failure();
    }

    // Keep padding values the same in the last dimension
    auto padVal = pad.getValues<int32_t>();

    std::vector<int32_t> paddingValues{0,
                                       0,
                                       static_cast<int>(padTop),
                                       static_cast<int>(padBottom),
                                       static_cast<int>(padLeft),
                                       static_cast<int>(padRight),
                                       padVal[{3, 0}],
                                       padVal[{3, 1}]};

    RankedTensorType paddingsType =
        RankedTensorType::get({4, 2}, rewriter.getI32Type());

    Value paddings = rewriter.create<TFL::ConstOp>(
        slice.getLoc(), DenseIntElementsAttr::get(paddingsType, paddingValues));

    auto paddedResultType =
        RankedTensorType::get({1, paddedHeight, paddedWidth, outputChannels},
                              utils::getValElementType(padOriginal.getInput()));

    auto padReplacement = rewriter.create<TFL::PadOp>(
        slice.getLoc(), paddedResultType, sliceReplacement, paddings);

    // replace slice with new slice -> new pad
    rewriter.replaceOp(slice, padReplacement.getOutput());

    return success();
  }
};

void OpSplit::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  auto &startOps = opSplitBottomOpsOption;
  auto &endOps = opSplitTopOpsOption;
  auto &numSplits = opSplitNumSplitsOption;

  // Check if the sizes of startOps, endOps, and numSplits are equal
  if (!(startOps.size() == endOps.size() &&
        endOps.size() == numSplits.size())) {
    // If they are not, emit an error message and signal pass failure
    func.emitError("start, end, and numSplits must be the same size");
    signalPassFailure();
    return;
  }

  if (numSplits.empty()) {
    int memoryThreshold = opSplitTargetSizeOption.getValue();
    // Initialize operation counter, tensor vectors, and size variables
    int opNum = 0;

    std::vector<mlir::Value> unconsumedTensors;
    std::vector<mlir::Value> newUnconsumedTensors;

    std::map<int, std::vector<size_t>> opSize;
    std::vector<size_t> sizeInfo;

    size_t currentTensorArenaSize;
    size_t inputSize;
    size_t outputSize;
    size_t residualSize;

    // Keep a pointer to the previous operation
    Operation *prevOp = nullptr;

    // Walk through each operation in the function
    func.walk([&](Operation *op) {
      // Ignore constant and quantized constant operations
      if (!op->hasTrait<OpTrait::IsTerminator>() &&
          !llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                     arith::ConstantOp>(op)) {

        // Helper function to compute the size of a tensor
        auto computeTensorSize = [](mlir::Type type) -> size_t {
          mlir::TensorType tensorType = type.cast<mlir::TensorType>();
          mlir::ArrayRef<int64_t> shape = tensorType.getShape();
          size_t tensorSize = 1;

          for (int64_t dim : shape) {
            tensorSize *= dim;
          }

          return tensorSize;
        };

        // Clear the contents of the vector
        newUnconsumedTensors.clear();
        // Iterate over unconsumed tensors and remove those consumed by the
        // current operation
        for (const mlir::Value &tensor : unconsumedTensors) {
          bool shouldRemove = false;
          for (mlir::Value::use_iterator it = tensor.use_begin(),
                                         e = tensor.use_end();
               it != e; ++it) {
            if ((*it).getOwner() == op) {
              shouldRemove = true;
              break;
            }
          }
          if (!shouldRemove) {
            newUnconsumedTensors.push_back(tensor);
          }
        }
        // Update unconsumed tensors with the new vector
        unconsumedTensors = newUnconsumedTensors;

        currentTensorArenaSize = 0;

        residualSize = 0;
        // Iterate over the unconsumed tensors and compute their sizes
        for (mlir::Value tensor : unconsumedTensors) {
          residualSize += computeTensorSize(tensor.getType());
          currentTensorArenaSize += computeTensorSize(tensor.getType());
        }

        inputSize = 0;
        // Iterate over the input operands and compute their sizes
        for (mlir::Value input : op->getOperands()) {
          if (!input.getType().isa<mlir::TensorType>()) {
            continue;
          }
          if (input.getDefiningOp() &&
              (input.getDefiningOp()->hasTrait<OpTrait::IsTerminator>() ||
               llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                         arith::ConstantOp>(input.getDefiningOp()))) {
            continue;
          }

          inputSize += computeTensorSize(input.getType());
          currentTensorArenaSize += computeTensorSize(input.getType());

          // If input tensor has more than one use and was created by the
          // previous operation, add it to unconsumed tensors
          if ((std::distance(input.use_begin(), input.use_end()) > 1) &&
              (input.getDefiningOp() == prevOp)) {
            unconsumedTensors.push_back(input);
          }
        }

        outputSize = 0;
        // Iterate over the output results and compute their sizes
        for (mlir::Value output : op->getResults()) {
          if (!output.getType().isa<mlir::TensorType>()) {
            continue;
          }
          if (output.getDefiningOp() &&
              (output.getDefiningOp()->hasTrait<OpTrait::IsTerminator>() ||
               llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                         arith::ConstantOp>(output.getDefiningOp()))) {
            continue;
          }
          outputSize += computeTensorSize(output.getType());
          currentTensorArenaSize += computeTensorSize(output.getType());
        }

        sizeInfo = {currentTensorArenaSize, inputSize, outputSize,
                    residualSize};
        opSize[opNum] = sizeInfo;

        // Increment operation counter
        opNum++;

        // Update the previous operation pointer
        prevOp = op;
      }
    });

    double size = 0;
    std::vector<int> aboveThreshold;
    std::vector<int> belowThreshold;
    bool crossedThreshold = false;

    for (auto it = opSize.rbegin(); it != opSize.rend(); ++it) {
      size = it->second[0];
      auto opId = it->first;
      if (size > memoryThreshold) {
        if (!crossedThreshold) {
          outputSize = it->second[2];
          // if 2 * output size is greater than the threshold,
          // concat will be greater than the threshold
          // so add the next op
          if (2 * outputSize > memoryThreshold) {
            aboveThreshold.push_back(opId + 1);
          } else {
            aboveThreshold.push_back(opId);
          }
          crossedThreshold = true;
        }
      } else {
        if (crossedThreshold) {
          belowThreshold.push_back(opId);
          crossedThreshold = false;
        }
      }
    }

    // If the first operation was above the threshold, add it, 0, to
    // belowThreshold
    if (crossedThreshold) {
      belowThreshold.push_back(0);
    }

    // adjust threshold trackers if size goes below threshold for only one
    // operation
    for (size_t i = 0; i < aboveThreshold.size(); ++i) {
      if (i > 0 && belowThreshold[i - 1] - aboveThreshold[i] <= 1) {
        aboveThreshold.erase(aboveThreshold.begin() + i);
        belowThreshold.erase(belowThreshold.begin() + i - 1);
        // Decrement the indices to account for the removed elements
        --i;
      }
    }

    // Clear the llvm::cl::list<int> containers first
    startOps.clear();
    endOps.clear();
    // Copy the elements from the std::vector<int> containers
    for (int value : aboveThreshold) {
      startOps.push_back(value);
    }
    for (int value : belowThreshold) {
      endOps.push_back(value);
    }
    for (size_t i = 0; i < startOps.size(); ++i) {
      numSplits.push_back(8);
    }

  } // if numSplits

  OpBuilder builder(func);
  for (int i = 0; i < startOps.size(); ++i) {
    int k = 0;
    func.walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::IsTerminator>() &&
          !llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                     arith::ConstantOp>(op)) {
        if (k == startOps[i]) {
          // If op is slice, just raise it, do not split it
          if (isa<TFL::SliceOp>(op)) {
            op->setAttr(opSplitLabel, builder.getUnitAttr());
            auto sliceOp = llvm::cast<TFL::SliceOp>(op);
            sliceOp.getInput().getDefiningOp()->setAttr(opSplitLabel,
                                                        builder.getUnitAttr());
          } else { // add label to insert slice under op later
            op->setAttr(opSplitLabelNumSplits,
                        builder.getI32IntegerAttr(numSplits[i]));
          }
        } else if (k < startOps[i] && k >= endOps[i]) {
          op->setAttr(opSplitLabel, builder.getUnitAttr());
        }
        k++;
      }
    });
  }

  RewritePatternSet patterns1(ctx);

  patterns1.insert<OpSplitHorizontalPattern<TFL::Conv2DOp>>(ctx);
  patterns1.insert<OpSplitHorizontalPattern<TFL::DepthwiseConv2DOp>>(ctx);
  patterns1.insert<OpSplitHorizontalPattern<TFL::AddOp>>(ctx);
  patterns1.insert<OpSplitHorizontalPattern<TFL::PadOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns1));

  RewritePatternSet patterns2(ctx);

  patterns2.insert<RaiseSliceHorizontalAddPattern>(ctx);
  patterns2.insert<RaiseSliceHorizontalPadPattern>(ctx);
  patterns2.insert<RaiseSliceHorizontalPattern<TFL::Conv2DOp>>(ctx);
  patterns2.insert<RaiseSliceHorizontalPattern<TFL::DepthwiseConv2DOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2));
} // void OpSplit::runOnOperation() {
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace mlir::xcore
