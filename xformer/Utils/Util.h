// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_UTIL_H
#define XFORMER_UTILS_UTIL_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::xcore::utils {

int getShapedTypeSize(ShapedType t);
bool hasSameShape(ShapedType type1, ShapedType type2);
size_t getTypeSize(Type type);
bool hasOnlyChannelPadding(DenseIntElementsAttr attr);
bool hasOnlySpatialPadding(DenseIntElementsAttr attr);

quant::UniformQuantizedType getQType(mlir::TypedValue<mlir::TensorType> tensor);
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

template <typename T> bool checkBinaryCompatibility(T op) {
  auto lhsType = op.getLhs().getType().template cast<ShapedType>();
  auto rhsType = op.getRhs().getType().template cast<ShapedType>();
  auto outputType = op.getOutput().getType().template cast<ShapedType>();

  // Check for invalid types and return
  // We don't currently handle the unusual case where both input shapes have
  // to be broadcasted. Either both input shapes must match the output or one
  // of the inputs has to be broadcasted.
  if (!hasSameShape(lhsType, rhsType) || !hasSameShape(lhsType, outputType)) {
    return false;
  }
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outputElemType = outputType.getElementType();

  if (!hasNBitSignedQType(lhsType) || !hasNBitSignedQType(rhsType) ||
      !hasNBitSignedQType(outputType)) {
    return false;
  }
  return true;
}
} // namespace mlir::xcore::utils

#endif // XFORMER_UTILS_UTIL_H
