// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_UTIL_H
#define XFORMER_UTILS_UTIL_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include <cstdint>

namespace mlir::xcore::utils {

int getShapedTypeSize(ShapedType t);
bool hasSameShape(ShapedType type1, ShapedType type2);
size_t getTypeSize(Type type);
SmallVector<int32_t, 8> getI32DimFromI64Dim(ArrayRef<int64_t> dims);
bool hasOnlyChannelPadding(DenseIntElementsAttr attr);
bool hasOnlySpatialPadding(DenseIntElementsAttr attr);

quant::UniformQuantizedType getQType(mlir::TypedValue<mlir::TensorType> tensor);

bool checkSliceNoOp(RankedTensorType inputType, RankedTensorType outputType);

template <int N> bool isNBitSignedQType(Type type) {
  return (type.template isa<quant::QuantizedType>() &&
          type.template cast<quant::QuantizedType>().isSigned() &&
          type.template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == N);
}

Type getValElementType(Value tensor);

ArrayRef<int64_t> getValShape(Value tensor);

template <typename To, typename From>
ArrayRef<To> castArrayRef(ArrayRef<From> ref) {
  std::vector<To> output;
  for (auto val : ref) {
    output.push_back(static_cast<To>(val));
  }
  return ArrayRef<To>(output);
}

template <typename T> bool checkBinaryCompatibility(T op) {
  auto lhsType = op.getLhs().getType().template cast<RankedTensorType>();
  auto rhsType = op.getRhs().getType().template cast<RankedTensorType>();
  auto outputType = op.getOutput().getType().template cast<RankedTensorType>();

  // Check for invalid types and return
  // We don't currently handle the unusual case where both input shapes have
  // to be broadcasted. Either both input shapes must match the output or one
  // of the inputs has to be broadcasted.
  if (!hasSameShape(rhsType, outputType) &&
      !hasSameShape(lhsType, outputType)) {
    return false;
  }
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outputElemType = outputType.getElementType();

  if (!isNBitSignedQType<8>(lhsElemType) ||
      !isNBitSignedQType<8>(rhsElemType) ||
      !isNBitSignedQType<8>(outputElemType)) {
    return false;
  }
  return true;
}

int mergeAxes(std::vector<int32_t> &begin, std::vector<int32_t> &size,
              std::vector<int32_t> &inShape, std::vector<int32_t> &outShape,
              int rank);

Value createShapeConstOp(PatternRewriter &rewriter, Location loc,
                         const SmallVector<int64_t, 4> &shapeVec);

LogicalResult
reshapeTransposeReshape(PatternRewriter &rewriter, Value tensor,
                        const SmallVector<int64_t, 4> &reshapeShape,
                        const SmallVector<int64_t, 4> &permVec,
                        const SmallVector<int64_t, 4> &origShape,
                        Value &result);

template <typename T = int64_t>
static SmallVector<T, 4> denseToVector(DenseIntElementsAttr permAttr) {
  SmallVector<T, 4> permVec;
  for (auto val : permAttr.getValues<int32_t>())
    permVec.push_back(static_cast<T>(val));
  return permVec;
}
} // namespace mlir::xcore::utils

#endif // XFORMER_UTILS_UTIL_H
