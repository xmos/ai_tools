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
static constexpr char opSplitLabelNumSplits[] = "opSplitLabelNumSplits";
static constexpr char opSplitLabelSavedNumSplits[] =
    "opSplitLabelSavedNumSplits";

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

  // all other uses of defining op must be eligible slices
  // we currently only opsplit ops with one result
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

  // no of eligible slices must be greater than or equal to set num of splits
  // If more slices, we should try to combine before raising
  if (!definingOp->hasAttr(opSplitLabelSavedNumSplits))
    return failure();
  auto attr = definingOp->getAttr(opSplitLabelSavedNumSplits);
  int numSplits = attr.cast<mlir::IntegerAttr>().getInt();
  if (numSplits != -1 && numEligibleSlices < numSplits) {
    return failure();
  } else {
    definingOp->setAttr(opSplitLabelSavedNumSplits,
                        rewriter.getI32IntegerAttr(-1));
  }

  return success();
}

LogicalResult combineSliceWithExisting(PatternRewriter &rewriter,
                                       TFL::SliceOp slice) {
  auto definingOp = slice.getInput().getDefiningOp();

  // all other uses of defining op must be eligible slices
  // we currently only opsplit ops with one result
  SmallVector<TFL::SliceOp> sliceOps;
  for (const mlir::OpOperand &use : definingOp->getResult(0).getUses()) {
    mlir::Operation *op = use.getOwner();
    if (auto sliceOp = dyn_cast_or_null<TFL::SliceOp>(op)) {
      // dont push current slice op
      if (sliceOp != slice) {
        sliceOps.push_back(sliceOp);
      }
    }
  }

  auto f = slice->getParentOfType<func::FuncOp>();
  int i;
  for (i = 0; i < sliceOps.size(); i++) {
    // if slice op matches with another op in list
    // remove current one and attach to that
    if (slice.getBegin() == sliceOps[i].getBegin() &&
        slice.getSize() == sliceOps[i].getSize()) {
      break;
    }
  }

  if (i < sliceOps.size()) {
    // slice.getOutput().replaceAllUsesWith(sliceOps[i]);
    // rewriter.eraseOp(slice);
    rewriter.replaceOp(slice, sliceOps[i].getOutput());
    return success();
  }
  // // replace slice with new slice -> new add
  // rewriter.replaceOp(fakeSlice, opReplacement->getResult(0));

  return failure();
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
    if (!slice.getInput().getDefiningOp() ||
        !isa<TFL::AddOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (failed(isRaisableSlice(rewriter, slice))) {
      return failure();
    }

    // combineslice with existing
    // go through all uses of defining op
    // find other slices and see if there is a match
    // if so, erase this slice and attach to that one, remove opsplitlabel from
    // attached new slice
    if (succeeded(combineSliceWithExisting(rewriter, slice))) {
      return success();
    }

    auto addOriginal = llvm::cast<TFL::AddOp>(slice.getInput().getDefiningOp());

    auto sliceOutShape = utils::getValShape(slice.getOutput());

    auto outputType =
        addOriginal.getOutput().getType().cast<RankedTensorType>();
    auto getSliceOp = [&](Value arg) -> Value {
      auto argType = arg.getType().cast<RankedTensorType>();
      auto newSlice = llvm::cast<TFL::SliceOp>(rewriter.clone(*slice));
      newSlice.setOperand(0, arg);
      RankedTensorType newSliceType =
          RankedTensorType::get(sliceOutShape, utils::getValElementType(arg));
      newSlice->getResult(0).setType(newSliceType);
      return newSlice;
    };

    // Create new slice for above adds
    auto sliceLHS = getSliceOp(addOriginal.getLhs());
    auto sliceRHS = getSliceOp(addOriginal.getRhs());
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

struct RaiseFakeSliceHorizontalPattern : public OpRewritePattern<FakeSliceOp> {
  using OpRewritePattern<FakeSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FakeSliceOp fakeSlice,
                                PatternRewriter &rewriter) const override {
    auto definingOp = fakeSlice.getInput().getDefiningOp();
    // If fakeslice does not have a defining op, remove it
    if (!definingOp) {
      rewriter.replaceOp(fakeSlice, fakeSlice->getResult(0));
      return success();
    }

    // no of fake slices must be equal to set num of splits
    if (!definingOp->hasAttr(opSplitLabelSavedNumSplits))
      return failure();
    auto attr = definingOp->getAttr(opSplitLabelSavedNumSplits);
    int numSplits = attr.cast<mlir::IntegerAttr>().getInt();
    auto beginAttr = fakeSlice->getAttr("begin").cast<mlir::ArrayAttr>();
    if (beginAttr.size() != numSplits) {
      return failure();
    }

    if (!dyn_cast_or_null<TFL::LogisticOp>(definingOp) &&
        !dyn_cast_or_null<TFL::FullyConnectedOp>(definingOp) &&
        !dyn_cast_or_null<TFL::ConstOp>(definingOp)) {
      return failure();
    }

    // Do not raise slice if op does not have op split label
    if (!(definingOp->hasAttr(opSplitLabel)))
      return failure();

    auto sliceOutShape = utils::getValShape(fakeSlice.getOutput());

    // Create new slice for above adds
    auto sliceReplacement = rewriter.clone(*fakeSlice);
    sliceReplacement->setOperand(0, definingOp->getOperand(0));
    sliceReplacement->getResult(0).setType(definingOp->getOperand(0).getType());
    auto opReplacement = rewriter.clone(*definingOp);
    opReplacement->setOperand(0, sliceReplacement->getResult(0));

    // replace slice with new slice -> new add
    rewriter.replaceOp(fakeSlice, opReplacement->getResult(0));

    return success();
  }
};

struct RaiseFakeSliceHorizontalMeanPattern
    : public OpRewritePattern<FakeSliceOp> {
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

    fakeSlice.dump();

    // Do not raise slice if op does not have op split label
    if (!(meanOriginal->hasAttr(opSplitLabel)))
      return failure();

    auto beginAttr = fakeSlice->getAttr("begin").cast<mlir::ArrayAttr>();
    auto sizeAttr = fakeSlice->getAttr("size").cast<mlir::ArrayAttr>();
    assert(beginAttr.size() == sizeAttr.size());

    SmallVector<Value> meanOps;

    for (int i = 0; i < beginAttr.size(); i++) {
      auto begin = beginAttr.getValue()[i].cast<mlir::DenseIntElementsAttr>();
      auto beginVector = std::vector<int32_t>{
          begin.getValues<int32_t>().begin(), begin.getValues<int32_t>().end()};

      auto size = sizeAttr.getValue()[i].cast<mlir::DenseIntElementsAttr>();
      auto sizeVector = std::vector<int32_t>{size.getValues<int32_t>().begin(),
                                             size.getValues<int32_t>().end()};

      // create slice and mean op
      auto sliceReplacement = createSliceOp(
          rewriter, fakeSlice.getLoc(), meanOriginal.getInput(), beginVector,
          sizeVector, utils::getValElementType(meanOriginal.getInput()));

      auto meanReplacement =
          llvm::cast<TFL::MeanOp>(rewriter.clone(*meanOriginal));
      meanReplacement.setOperand(0, sliceReplacement);

      meanOps.push_back(meanReplacement.getResult());
    }
    // create concat and final mean op
    RankedTensorType newOutputType = RankedTensorType::get(
        {1, 4, 1, 16}, utils::getValElementType(meanOriginal.getOutput()));
    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        meanOriginal.getLoc(), newOutputType, meanOps, /*axis=*/1, "NONE");

    auto meanReplacement =
        llvm::cast<TFL::MeanOp>(rewriter.clone(*meanOriginal));
    meanReplacement.setOperand(0, newConcatOp);

    // auto sliceOutShape = utils::getValShape(fakeSlice.getOutput());

    // // Create new slice for above adds
    // auto sliceReplacement = rewriter.clone(*fakeSlice);
    // sliceReplacement->setOperand(0, definingOp.getOperand());
    // sliceReplacement->getResult(0).setType(definingOp.getResult().getType());

    // auto opReplacement = rewriter.clone(*definingOp);
    // opReplacement->setOperand(0, sliceReplacement->getResult(0));

    // // replace slice with new slice -> new add
    rewriter.replaceOp(fakeSlice, meanReplacement->getResult(0));

    return success();
  }
};

struct RaiseSliceHorizontalMulPattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto f = slice->getParentOfType<func::FuncOp>();
    // If slice does not have a defining op, return failure
    if (!slice.getInput().getDefiningOp() ||
        !isa<TFL::MulOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (failed(isRaisableSlice(rewriter, slice))) {
      return failure();
    }

    auto addOriginal = llvm::cast<TFL::MulOp>(slice.getInput().getDefiningOp());

    DenseElementsAttr beginAttr, sizeAttr;
    if (!matchPattern(slice.getBegin(), m_Constant(&beginAttr))) {
      return failure();
    }
    if (!matchPattern(slice.getSize(), m_Constant(&sizeAttr))) {
      return failure();
    }

    auto sliceOutShape = utils::getValShape(slice.getOutput());
    auto addReplacement = llvm::cast<TFL::MulOp>(rewriter.clone(*addOriginal));
    RankedTensorType addReplacementType = RankedTensorType::get(
        sliceOutShape, utils::getValElementType(addOriginal.getOutput()));
    addReplacement->getResult(0).setType(addReplacementType);

    auto outputType =
        addOriginal.getOutput().getType().cast<RankedTensorType>();
    auto getSliceOp = [&](int argNo, Value arg) -> Value {
      auto argType = arg.getType().cast<RankedTensorType>();
      if (utils::hasSameShape(argType, outputType)) {
        rewriter.setInsertionPoint(addReplacement);
        auto newSlice = llvm::cast<TFL::SliceOp>(rewriter.clone(*slice));
        newSlice.setOperand(0, arg);
        RankedTensorType newSliceType =
            RankedTensorType::get(sliceOutShape, utils::getValElementType(arg));
        newSlice->getResult(0).setType(newSliceType);
        return newSlice;
      } else {
        auto fakeSlice = dyn_cast_or_null<FakeSliceOp>(arg.getDefiningOp());
        if (!fakeSlice) {
          rewriter.setInsertionPoint(addOriginal);
          auto newFsOp =
              rewriter.create<FakeSliceOp>(arg.getLoc(), arg.getType(), arg);

          llvm::SmallVector<mlir::Attribute> beginVals;
          beginVals.push_back(beginAttr);
          newFsOp->setAttr("begin", rewriter.getArrayAttr(beginVals));

          llvm::SmallVector<mlir::Attribute> sizeVals;
          sizeVals.push_back(sizeAttr);
          newFsOp->setAttr("size", rewriter.getArrayAttr(sizeVals));

          auto addReplacement =
              llvm::cast<TFL::MulOp>(rewriter.clone(*addOriginal));
          addReplacement.setOperand(argNo, newFsOp);
          addOriginal.getOutput().replaceAllUsesWith(addReplacement);
          rewriter.eraseOp(addOriginal);
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

    // Create new slice for above adds
    auto sliceLHS = getSliceOp(0, addOriginal.getLhs());
    auto sliceRHS = getSliceOp(1, addOriginal.getRhs());
    // auto addReplacement =
    // llvm::cast<TFL::MulOp>(rewriter.clone(*addOriginal2));
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
    // If slice does not have a defining op, return failure
    if (!slice.getInput().getDefiningOp() ||
        !isa<ConvOp>(slice.getInput().getDefiningOp())) {
      return failure();
    }

    if (failed(isRaisableSlice(rewriter, slice))) {
      return failure();
    }

    // combineslice with existing
    // go through all uses of defining op
    // find other slices and see if there is a match
    // if so, erase this slice and attach to that one, remove opsplitlabel from
    // attached new slice
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
            op->setAttr(opSplitLabelSavedNumSplits,
                        builder.getI32IntegerAttr(numSplits[i]));
          }
        } else if (k < startOps[i] && k >= endOps[i]) {
          op->setAttr(opSplitLabel, builder.getUnitAttr());
          op->setAttr(opSplitLabelSavedNumSplits,
                      builder.getI32IntegerAttr(numSplits[i]));
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
  patterns2.insert<RaiseSliceHorizontalMulPattern>(ctx);
  patterns2.insert<RaiseSliceHorizontalPadPattern>(ctx);
  patterns2.insert<RaiseSliceHorizontalPattern<TFL::Conv2DOp>>(ctx);
  patterns2.insert<RaiseSliceHorizontalPattern<TFL::DepthwiseConv2DOp>>(ctx);

  patterns2.insert<RaiseFakeSliceHorizontalPattern>(ctx);
  patterns2.insert<RaiseFakeSliceHorizontalMeanPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2));

} // void OpSplit::runOnOperation() {
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace mlir::xcore
