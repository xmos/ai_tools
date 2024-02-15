// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_UTIL_H
#define XFORMER_UTILS_UTIL_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace xcore {
namespace utils {

int getShapedTypeSize(ShapedType t);

LogicalResult hasSameShape(ShapedType type1, ShapedType type2);

size_t getTypeSize(Type type);

template <typename T>
bool checkSliceNoOp(T beginValues, T sizeValues, RankedTensorType type) {
  const int rank = type.getRank();
  bool isNoOp = true;
  for (int i = 0; i < rank; i++) {
    if (beginValues[i] != 0 || sizeValues[i] != type.getShape()[i]) {
      isNoOp = false;
      break;
    }
  }
  return isNoOp;
}

template <int N = 8> bool hasNBitSignedQType(Type type) {
  return (type.template isa<quant::QuantizedType>() &&
          type.template cast<quant::QuantizedType>().isSigned() &&
          type.template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == N);
}
} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_UTIL_H
