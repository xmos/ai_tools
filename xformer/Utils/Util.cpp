// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/Util.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::xcore::utils {

size_t getTypeSize(Type type) {
  if (auto quantType = type.dyn_cast<quant::QuantizedType>()) {
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
  int sizeInBytes = getTypeSize(t.getElementType());

  llvm::ArrayRef<int64_t> shape = t.getShape();
  // Handle dynamic shapes
  for (auto &dim : shape) {
    sizeInBytes *= (ShapedType::isDynamic(dim) ? 1 : dim);
  }

  return sizeInBytes;
}

quant::UniformQuantizedType
getQType(mlir::TypedValue<mlir::TensorType> tensor) {
  return tensor.getType()
      .cast<RankedTensorType>()
      .getElementType()
      .cast<quant::UniformQuantizedType>();
}

bool hasSameShape(ShapedType type1, ShapedType type2) {
  llvm::ArrayRef<int64_t> shape1 = type1.getShape();
  llvm::ArrayRef<int64_t> shape2 = type2.getShape();

  if (shape1.size() != shape2.size()) {
    return false;
  }

  // Handle dynamic shapes
  for (int i = 0; i < shape1.size(); i++) {
    int d1 = (ShapedType::isDynamic(shape1[i]) ? 1 : shape1[i]);
    int d2 = (ShapedType::isDynamic(shape2[i]) ? 1 : shape2[i]);
    if (d1 != d2) {
      return false;
    }
  }

  return true;
}

bool hasOnlyChannelPadding(DenseIntElementsAttr attr) {
  if (attr.getNumElements() != 8)
    return false;
  auto values = attr.getValues<int32_t>();
  return values[{0, 0}] == 0 && values[{0, 1}] == 0 && values[{1, 0}] == 0 &&
         values[{1, 1}] == 0 && values[{2, 0}] == 0 && values[{2, 1}] == 0;
}

bool hasOnlySpatialPadding(DenseIntElementsAttr attr) {
  if (attr.getNumElements() != 8)
    return false;
  auto values = attr.getValues<int32_t>();
  return values[{0, 0}] == 0 && values[{0, 1}] == 0 && values[{3, 0}] == 0 &&
         values[{3, 1}] == 0;
}

Type getValElementType(Value tensor) {
  return tensor.getType().template cast<RankedTensorType>().getElementType();
}

ArrayRef<int64_t> getValShape(Value tensor) {
  return tensor.getType().template cast<RankedTensorType>().getShape();
}

bool checkSliceNoOp(RankedTensorType inputType, RankedTensorType outputType) {
  const int rank = inputType.getRank();
  bool isNoOp = true;
  for (int i = 0; i < rank; i++) {
    if (inputType.getDimSize(i) != outputType.getDimSize(i)) {
      isNoOp = false;
      break;
    }
  }
  return isNoOp;
}
} // namespace mlir::xcore::utils
