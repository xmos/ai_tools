// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/Util.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace xcore {
namespace utils {

int getShapedTypeSize(ShapedType t) {
  int sizeInBytes;
  if (t.getElementType().isa<quant::QuantizedType>()) {
    // we only support QI8
    sizeInBytes = 1;
  } else {
    sizeInBytes = t.getElementType().getIntOrFloatBitWidth() / CHAR_BIT;
  }

  llvm::ArrayRef<int64_t> shape = t.getShape();
  // Handle dynamic shapes
  for (auto &dim : shape) {
    sizeInBytes *= (ShapedType::isDynamic(dim) ? 1 : dim);
  }

  return sizeInBytes;
}

LogicalResult hasSameShape(ShapedType type1, ShapedType type2) {
  llvm::ArrayRef<int64_t> shape1 = type1.getShape();
  llvm::ArrayRef<int64_t> shape2 = type2.getShape();

  if (shape1.size() != shape2.size()) {
    return failure();
  }

  // Handle dynamic shapes
  for (int i = 0; i < shape1.size(); i++) {
    int d1 = (ShapedType::isDynamic(shape1[i]) ? 1 : shape1[i]);
    int d2 = (ShapedType::isDynamic(shape2[i]) ? 1 : shape2[i]);
    if (d1 != d2) {
      return failure();
    }
  }

  return success();
}

} // namespace utils
} // namespace xcore
} // namespace mlir
