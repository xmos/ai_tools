// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_UTIL_H
#define XFORMER_UTILS_UTIL_H

#include "mlir/IR/BuiltinAttributes.h" // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace xcore {
namespace utils {

int getShapedTypeSize(ShapedType t);

LogicalResult hasSameShape(ShapedType type1, ShapedType type2);

inline bool IsAllOnesConstant(Attribute value) {
  auto values = value.cast<DenseElementsAttr>().getValues<int32_t>();
  return !std::any_of(values.begin(), values.end(),
                      [](int32_t element_value) { return element_value != 1; });
}

// Utility function to get the offset between two dense attribute values.
inline TypedAttr GetOffSet(Attribute begin, Attribute end) {
  auto begin_values = begin.cast<DenseElementsAttr>().getValues<int32_t>();
  auto end_values = end.cast<DenseElementsAttr>().getValues<int32_t>();

  SmallVector<int32_t> offsets;
  if (begin_values.size() == end_values.size()) {
    for (size_t i = 0; i < begin_values.size(); ++i) {
      offsets.push_back(end_values[i] - begin_values[i]);
    }
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get({static_cast<int>(offsets.size())},
                            mlir::IntegerType::get(begin.getContext(), 32)),
      ArrayRef(offsets));
}

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_UTIL_H
