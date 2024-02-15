// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/Util.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::xcore::utils {

size_t getTypeSize(Type type) {
  if (auto quantType = type.dyn_cast<quant::UniformQuantizedType>()) {
    return quantType.getStorageType().getIntOrFloatBitWidth() / 8;
  } else if (auto floatType = type.dyn_cast<FloatType>()) {
    return floatType.getWidth() / 8;
  } else if (auto intType = type.dyn_cast<IntegerType>()) {
    return intType.getWidth() / 8;
  } else {
    llvm_unreachable("Unsupported type");
  }
  return 0;
}

int getShapedTypeSize(ShapedType t) {
  int sizeInBytes;
  if (auto quantType = t.getElementType().dyn_cast<quant::QuantizedType>()) {
    sizeInBytes = quantType.getStorageTypeIntegralWidth() / CHAR_BIT;
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

} // namespace mlir::xcore::utils
