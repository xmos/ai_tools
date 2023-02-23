// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// Utility predicates that are shared by multiple passes.

// Check that bytes per pixel is a multiple of n
class HasMultipleOfNBytesPerPixel<int n>
    : Constraint<CPred<
          "$0.getType().cast<ShapedType>().getDimSize(3) % " #n #" == 0">>;

// Spatial padding
def HasNoPaddingForBatchAndChannel
    : Constraint<CPred<
          "$0.cast<DenseIntElementsAttr>().getNumElements() == 8 && "
          "($0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 0}] == 0 "
          "&&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 1}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 0}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 1}] == "
          "0)">>;

// Only 3 to 4 channel padding
def HasOnlyPadding3To4ForChannel
    : Constraint<CPred<
          "$0.cast<DenseIntElementsAttr>().getNumElements() == 8 && "
          "($0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 0}] == 0 "
          "&&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 1}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{1, 0}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{1, 1}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{2, 0}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{2, 1}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 0}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 1}] == "
          "1)">>;
