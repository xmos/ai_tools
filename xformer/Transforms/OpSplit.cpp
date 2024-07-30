//  Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir::xcore {

namespace {
static constexpr char opSplitLabel[] = "opSplitLabel";
static constexpr char opSplitLabelStartSplits[] = "opSplitLabelStartSplits";
static constexpr char opSplitLabelNumSplits[] = "opSplitLabelNumSplits";

// OpSplit
struct OpSplit : public PassWrapper<OpSplit, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSplit)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
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

LogicalResult isRaisableSlice(PatternRewriter &rewriter, TFL::SliceOp slice) {
  // Only raise slices that have been inserted with op split pass
  if (!slice->hasAttr(opSplitLabel))
    return failure();

  auto definingOp = slice.getInput().getDefiningOp();
  // Do not raise slice if defining op does not have op split label
  if (!definingOp->hasAttr(opSplitLabel))
    return failure();

  // All other uses of defining op must be eligible slices
  // We currently only opsplit ops with one result
  int numEligibleSlices = 0;
  for (const mlir::OpOperand &use : definingOp->getResult(0).getUses()) {
    mlir::Operation *op = use.getOwner();
    if (auto sliceOp = dyn_cast_or_null<TFL::SliceOp>(op)) {
      if (!sliceOp->hasAttr(opSplitLabel)) {
        return failure();
      } else {
        numEligibleSlices++;
      }
    } else {
      return failure();
    }
  }

  // No of eligible slices must be greater than or equal to set num of splits
  // If there are more slices, we should try to combine before raising
  if (!definingOp->hasAttr(opSplitLabelNumSplits))
    return failure();

  auto attr = definingOp->getAttr(opSplitLabelNumSplits);
  int numSplits = attr.cast<mlir::IntegerAttr>().getInt();
  if (numSplits != -1 && numEligibleSlices < numSplits) {
    return failure();
  } else {
    definingOp->setAttr(opSplitLabelNumSplits, rewriter.getI32IntegerAttr(-1));
  }

  return success();
}

LogicalResult combineSliceWithExisting(PatternRewriter &rewriter,
                                       TFL::SliceOp slice) {
  auto definingOp = slice.getInput().getDefiningOp();

  // All other uses of defining op must be slices
  // We currently only opsplit ops with one result
  SmallVector<TFL::SliceOp> sliceOps;
  for (const mlir::OpOperand &use : definingOp->getResult(0).getUses()) {
    mlir::Operation *op = use.getOwner();
    if (auto sliceOp = dyn_cast_or_null<TFL::SliceOp>(op)) {
      // We only support slices on height dimension
      // Slice must have rank 4 and dim 0, 2, 3 must be the same for input and
      // output
      auto inType = sliceOp.getInput().getType().cast<ShapedType>();
      auto outType = sliceOp.getOutput().getType().cast<ShapedType>();
      if (!inType.getRank() == 4 ||
          inType.getDimSize(0) != outType.getDimSize(0) ||
          inType.getDimSize(2) != outType.getDimSize(2) ||
          inType.getDimSize(3) != outType.getDimSize(3)) {
        return failure();
      }
      // Dont push current slice op
      if (sliceOp != slice) {
        sliceOps.push_back(sliceOp);
      }
    }
  }

  auto f = slice->getParentOfType<func::FuncOp>();

  // Get begin index for slice
  DenseElementsAttr attr;
  int sliceBegin, sliceSize, candidateBegin, candidateSize;
  if (!matchPattern(slice.getBegin(), m_Constant(&attr))) {
    return failure();
  }
  sliceBegin = attr.getValues<int32_t>()[1];
  if (!matchPattern(slice.getSize(), m_Constant(&attr))) {
    return failure();
  }
  sliceSize = attr.getValues<int32_t>()[1];

  int i;
  for (i = 0; i < sliceOps.size(); i++) {
    // If slice op matches with another op in list,
    // remove current one and attach to that
    // Only need to consider height dimension as we only slice on that
    if (!matchPattern(sliceOps[i].getBegin(), m_Constant(&attr))) {
      return failure();
    }
    candidateBegin = attr.getValues<int32_t>()[1];
    if (!matchPattern(sliceOps[i].getSize(), m_Constant(&attr))) {
      return failure();
    }
    candidateSize = attr.getValues<int32_t>()[1];

    if (sliceBegin >= candidateBegin &&
        sliceBegin + sliceSize <= candidateBegin + candidateSize) {
      break;
    }
  }

  if (i < sliceOps.size()) {
    // Slice can be removed
    if (sliceBegin == candidateBegin && sliceSize == candidateSize) {
      rewriter.replaceOp(slice, sliceOps[i].getOutput());
    } else {
      // Create new slice
      if (!matchPattern(sliceOps[i].getBegin(), m_Constant(&attr))) {
        return failure();
      }
      int32_t newBeginAttr[4] = {
          attr.getValues<int32_t>()[0], sliceBegin - candidateBegin,
          attr.getValues<int32_t>()[2], attr.getValues<int32_t>()[3]};
      if (!matchPattern(slice.getSize(), m_Constant(&attr))) {
        return failure();
      }
      // Same as slice size attr
      int32_t newSizeAttr[4] = {
          attr.getValues<int32_t>()[0], attr.getValues<int32_t>()[1],
          attr.getValues<int32_t>()[2], attr.getValues<int32_t>()[3]};
      auto newSlice = createSliceOp(
          rewriter, sliceOps[i].getLoc(), sliceOps[i], newBeginAttr,
          newSizeAttr, slice.getOutput().getType().getElementType());
      newSlice->removeAttr(opSplitLabel);
      rewriter.replaceOp(slice, newSlice.getOutput());
    }
    return success();
  }
  return failure();
}

template <typename TargetOp>
struct OpSplitPattern : public OpRewritePattern<TargetOp> {
  using OpRewritePattern<TargetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TargetOp targetOp,
                                PatternRewriter &rewriter) const override {
    // Do not split ops already split
    // Only start splitting if the label is present
    if (!(targetOp->hasAttr(opSplitLabelStartSplits)))
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

template <typename TargetOp>
struct RaiseFakeSlicePattern : public OpRewritePattern<FakeSliceOp> {
  using OpRewritePattern<FakeSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FakeSliceOp fakeSlice,
                                PatternRewriter &rewriter) const override {
    auto definingOp = fakeSlice.getInput().getDefiningOp();
    // If fakeslice is only defined by const ops, remove it as we have reached
    // the top
    if (dyn_cast_or_null<TFL::ConstOp>(definingOp) ||
        dyn_cast_or_null<TFL::QConstOp>(definingOp)) {
      rewriter.replaceOp(fakeSlice, fakeSlice->getOperand(0));
      return success();
    }

    // No of fake slices must be equal to set num of splits
    if (!definingOp->hasAttr(opSplitLabelNumSplits))
      return failure();
    auto attr = definingOp->getAttr(opSplitLabelNumSplits);
    int numSplits = attr.cast<mlir::IntegerAttr>().getInt();
    auto beginAttr = fakeSlice->getAttr("begin").cast<mlir::ArrayAttr>();
    if (beginAttr.size() != numSplits) {
      return failure();
    }

    if (!dyn_cast_or_null<TargetOp>(definingOp)) {
      return failure();
    }

    // Do not raise fake slice if op does not have op split label
    if (!(definingOp->hasAttr(opSplitLabel)))
      return failure();

    auto sliceOutShape = utils::getValShape(fakeSlice.getOutput());

    // Create new fake slice above op
    auto sliceReplacement = rewriter.clone(*fakeSlice);
    sliceReplacement->setOperand(0, definingOp->getOperand(0));
    sliceReplacement->getResult(0).setType(definingOp->getOperand(0).getType());
    auto opReplacement = rewriter.clone(*definingOp);
    opReplacement->setOperand(0, sliceReplacement->getResult(0));

    // replace fakeslice with new fakeslice -> op
    rewriter.replaceOp(fakeSlice, opReplacement->getResult(0));

    return success();
  }
};

template <typename TargetOp>
struct RaiseFakeSliceConstPattern : public OpRewritePattern<FakeSliceOp> {
  using OpRewritePattern<FakeSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FakeSliceOp fakeSlice,
                                PatternRewriter &rewriter) const override {
    auto definingOp = fakeSlice.getInput().getDefiningOp();
    // If fakeslice is only defined by const ops, remove it as we have reached
    // the top
    if (dyn_cast_or_null<TargetOp>(definingOp)) {
      rewriter.replaceOp(fakeSlice, fakeSlice->getOperand(0));
      return success();
    }
    return failure();
  }
};

struct RaiseFakeSliceToSliceMeanPattern : public OpRewritePattern<FakeSliceOp> {
  using OpRewritePattern<FakeSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FakeSliceOp fakeSlice,
                                PatternRewriter &rewriter) const override {
    // If slice does not have a defining op, return failure
    if (!(fakeSlice.getInput().getDefiningOp())) {
      return failure();
    }

    auto meanOriginal =
        dyn_cast_or_null<TFL::MeanOp>(fakeSlice.getInput().getDefiningOp());
    if (!meanOriginal) {
      return failure();
    }

    // Do not raise slice if op does not have op split label
    if (!(meanOriginal->hasAttr(opSplitLabel)))
      return failure();

    auto beginAttr = fakeSlice->getAttr("begin").cast<mlir::ArrayAttr>();
    auto sizeAttr = fakeSlice->getAttr("size").cast<mlir::ArrayAttr>();
    assert(beginAttr.size() == sizeAttr.size());

    // We only support all equal slices at the moment
    assert(sizeAttr.size() > 1);
    auto size = sizeAttr.getValue()[0].cast<mlir::DenseIntElementsAttr>();
    for (int i = 1; i < sizeAttr.size(); i++) {
      auto size2 = sizeAttr.getValue()[i].cast<mlir::DenseIntElementsAttr>();
      if (size != size2) {
        return failure();
      }
    }

    // For each begin and size attr, create slice and corresponding mean op
    SmallVector<Value> meanOps;
    for (int i = 0; i < beginAttr.size(); i++) {
      auto begin = beginAttr.getValue()[i].cast<mlir::DenseIntElementsAttr>();
      auto beginVector = std::vector<int32_t>{
          begin.getValues<int32_t>().begin(), begin.getValues<int32_t>().end()};

      auto size = sizeAttr.getValue()[i].cast<mlir::DenseIntElementsAttr>();
      auto sizeVector = std::vector<int32_t>{size.getValues<int32_t>().begin(),
                                             size.getValues<int32_t>().end()};

      // Create slice and mean op
      auto sliceReplacement = createSliceOp(
          rewriter, fakeSlice.getLoc(), meanOriginal.getInput(), beginVector,
          sizeVector, utils::getValElementType(meanOriginal.getInput()));

      auto meanReplacement =
          llvm::cast<TFL::MeanOp>(rewriter.clone(*meanOriginal));
      meanReplacement.setOperand(0, sliceReplacement);

      meanOps.push_back(meanReplacement.getResult());
    }
    // Create concat and final mean op
    auto meanOutShape = utils::getValShape(meanOriginal.getOutput());
    RankedTensorType newOutputType = RankedTensorType::get(
        {1, static_cast<long long>(beginAttr.size()), 1, meanOutShape[3]},
        utils::getValElementType(meanOriginal.getOutput()));
    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        meanOriginal.getLoc(), newOutputType, meanOps, /*axis=*/1, "NONE");

    auto meanReplacement =
        llvm::cast<TFL::MeanOp>(rewriter.clone(*meanOriginal));
    meanReplacement.setOperand(0, newConcatOp);

    // Replace fake slice with new slices -> mean ops -> concat -> mean
    rewriter.replaceOp(fakeSlice, meanReplacement->getResult(0));

    return success();
  }
};

template <typename BinaryOp>
struct RaiseSliceBinaryPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto f = slice->getParentOfType<func::FuncOp>();
    // If slice does not have a defining op, return failure
    if (!slice.getInput().getDefiningOp() ||
        !isa<BinaryOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (failed(isRaisableSlice(rewriter, slice))) {
      return failure();
    }

    if (succeeded(combineSliceWithExisting(rewriter, slice))) {
      return success();
    }

    auto opOriginal = llvm::cast<BinaryOp>(slice.getInput().getDefiningOp());

    DenseElementsAttr beginAttr, sizeAttr;
    if (!matchPattern(slice.getBegin(), m_Constant(&beginAttr))) {
      return failure();
    }
    if (!matchPattern(slice.getSize(), m_Constant(&sizeAttr))) {
      return failure();
    }

    auto sliceOutShape = utils::getValShape(slice.getOutput());
    auto opReplacement = llvm::cast<BinaryOp>(rewriter.clone(*opOriginal));
    RankedTensorType opReplacementType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(opOriginal.getOutput()));
    opReplacement->getResult(0).setType(opReplacementType);

    auto outputType =
        opOriginal.getOutput().getType().template cast<RankedTensorType>();
    auto getSliceOp = [&](int argNo, Value arg) -> Value {
      auto argType = arg.getType().cast<RankedTensorType>();
      if (utils::hasSameShape(argType, outputType)) {
        rewriter.setInsertionPoint(opReplacement);
        auto newSlice = llvm::cast<TFL::SliceOp>(rewriter.clone(*slice));
        newSlice.setOperand(0, arg);
        RankedTensorType newSliceType =
            RankedTensorType::get(sliceOutShape, utils::getValElementType(arg));
        newSlice->getResult(0).setType(newSliceType);
        return newSlice;
      } else {
        auto fakeSlice = dyn_cast_or_null<FakeSliceOp>(arg.getDefiningOp());
        if (!fakeSlice) {
          rewriter.setInsertionPoint(opOriginal);
          auto newFsOp =
              rewriter.create<FakeSliceOp>(arg.getLoc(), arg.getType(), arg);

          llvm::SmallVector<mlir::Attribute> beginVals;
          beginVals.push_back(beginAttr);
          newFsOp->setAttr("begin", rewriter.getArrayAttr(beginVals));

          llvm::SmallVector<mlir::Attribute> sizeVals;
          sizeVals.push_back(sizeAttr);
          newFsOp->setAttr("size", rewriter.getArrayAttr(sizeVals));

          auto opReplacement =
              llvm::cast<BinaryOp>(rewriter.clone(*opOriginal));
          opReplacement.setOperand(argNo, newFsOp);
          opOriginal.getOutput().replaceAllUsesWith(opReplacement);
          rewriter.eraseOp(opOriginal);
          return newFsOp;
        } else {
          auto begin = fakeSlice->getAttr("begin").cast<mlir::ArrayAttr>();
          llvm::SmallVector<mlir::Attribute> beginVals = {
              begin.getValue().begin(), begin.getValue().end()};
          beginVals.push_back(beginAttr);
          fakeSlice->setAttr("begin", rewriter.getArrayAttr(beginVals));

          auto size = fakeSlice->getAttr("size").cast<mlir::ArrayAttr>();
          llvm::SmallVector<mlir::Attribute> sizeVals = {
              size.getValue().begin(), size.getValue().end()};
          sizeVals.push_back(sizeAttr);
          fakeSlice->setAttr("size", rewriter.getArrayAttr(sizeVals));
        }
        return fakeSlice;
      }
    };

    // Create new slices for above op
    auto sliceLHS = getSliceOp(0, opOriginal.getLhs());
    auto sliceRHS = getSliceOp(1, opOriginal.getRhs());
    opReplacement.setOperand(0, sliceLHS);
    opReplacement.setOperand(1, sliceRHS);

    // replace slice with new slice -> new op
    rewriter.replaceOp(slice, opReplacement.getOutput());

    return success();
  }
};

template <typename ConvOp>
struct RaiseSlicePattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    // If slice does not have a defining op, return failure
    if (!slice.getInput().getDefiningOp() ||
        !isa<ConvOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (failed(isRaisableSlice(rewriter, slice))) {
      return failure();
    }

    if (succeeded(combineSliceWithExisting(rewriter, slice))) {
      return success();
    }

    // Get data from conv needed to raise slice
    auto convOriginal = llvm::cast<ConvOp>(slice.getInput().getDefiningOp());

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

struct RaiseSlicePadPattern : public OpRewritePattern<TFL::SliceOp> {
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

    int64_t padVertical, pad;
    int64_t padTop, padBottom, padLeft, padRight;

    padVertical = outputHeight - inputHeight;
    pad = outputWidth - inputWidth;

    padTop = padVertical / 2;
    padBottom = padVertical - padTop;
    padLeft = pad / 2;
    padRight = pad - padLeft;

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

    DenseIntElementsAttr padAttr;
    if (!matchPattern(padOriginal.getPadding(), m_Constant(&padAttr))) {
      return failure();
    }

    // Keep padding values the same in the last dimension
    auto padVal = padAttr.getValues<int32_t>();

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
            op->setAttr(opSplitLabelStartSplits, builder.getUnitAttr());
            op->setAttr(opSplitLabelNumSplits,
                        builder.getI32IntegerAttr(numSplits[i]));
          }
        } else if (k < startOps[i] && k >= endOps[i]) {
          op->setAttr(opSplitLabel, builder.getUnitAttr());
          op->setAttr(opSplitLabelNumSplits,
                      builder.getI32IntegerAttr(numSplits[i]));
        }
        k++;
      }
    });
  }

  RewritePatternSet patterns1(ctx);

  patterns1.insert<OpSplitPattern<TFL::Conv2DOp>>(ctx);
  patterns1.insert<OpSplitPattern<TFL::DepthwiseConv2DOp>>(ctx);
  patterns1.insert<OpSplitPattern<TFL::AddOp>>(ctx);
  patterns1.insert<OpSplitPattern<TFL::MulOp>>(ctx);
  patterns1.insert<OpSplitPattern<TFL::PadOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns1));

  RewritePatternSet patterns2(ctx);
  // We are restricting pattern matching to only move slices above when they
  // have reached the number of set splits. This means many pass iterations
  // would fail as they don't meet the criteria. We increase the maxIterations
  // count here so that more iterations are tried before the rewriter decides
  // failure.
  GreedyRewriteConfig config;
  config.maxIterations = 50;

  patterns2.insert<RaiseSliceBinaryPattern<TFL::AddOp>>(ctx);
  patterns2.insert<RaiseSliceBinaryPattern<TFL::MulOp>>(ctx);
  patterns2.insert<RaiseSlicePadPattern>(ctx);
  patterns2.insert<RaiseSlicePattern<TFL::Conv2DOp>>(ctx);
  patterns2.insert<RaiseSlicePattern<TFL::DepthwiseConv2DOp>>(ctx);

  patterns2.insert<RaiseFakeSlicePattern<TFL::FullyConnectedOp>>(ctx);
  patterns2.insert<RaiseFakeSlicePattern<TFL::LogisticOp>>(ctx);
  patterns2.insert<RaiseFakeSliceConstPattern<TFL::ConstOp>>(ctx);
  patterns2.insert<RaiseFakeSliceConstPattern<TFL::QConstOp>>(ctx);
  patterns2.insert<RaiseFakeSliceToSliceMeanPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2), config);

} // void OpSplit::runOnOperation() {
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace mlir::xcore
