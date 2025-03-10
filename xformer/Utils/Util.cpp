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

SmallVector<int32_t, 8> getI32DimFromI64Dim(ArrayRef<int64_t> dims) {
  SmallVector<int32_t, 8> output_shape_values;
  for (auto dim : dims) {
    output_shape_values.push_back(
        ShapedType::isDynamic(dim) ? -1 : static_cast<int32_t>(dim));
  }
  return output_shape_values;
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
  if (rank != outputType.getRank()) {
    return false;
  }
  bool isNoOp = true;
  for (int i = 0; i < rank; i++) {
    if (inputType.getDimSize(i) != outputType.getDimSize(i)) {
      isNoOp = false;
      break;
    }
  }
  return isNoOp;
}

int mergeAxes(std::vector<int32_t> &begin, std::vector<int32_t> &size,
              std::vector<int32_t> &inShape, std::vector<int32_t> &outShape,
              int rank) {

  for (int i = rank - 1; i > 0; i--) {
    while ((inShape[i] == outShape[i]) && (i > 0)) {
      const int mul = inShape[i];
      inShape[i - 1] *= mul;
      outShape[i - 1] *= mul;
      begin[i - 1] *= mul;
      size[i - 1] *= mul;
      inShape.erase(inShape.begin() + i);
      outShape.erase(outShape.begin() + i);
      begin.erase(begin.begin() + i);
      size.erase(size.begin() + i);
      rank -= 1;
      i -= 1;
    }
  }
  if ((inShape[0] == 1) && (outShape[0] == 1)) {
    inShape.erase(inShape.begin());
    outShape.erase(outShape.begin());
    begin.erase(begin.begin());
    size.erase(size.begin());
    rank -= 1;
  }
  return rank;
}

// Creates a constant op for a shape vector.
Value createShapeConstOp(PatternRewriter &rewriter, Location loc,
                         const SmallVector<int64_t, 4> &shapeVec) {
  SmallVector<int32_t, 4> shapeVecI32;
  for (auto val : shapeVec) {
    shapeVecI32.push_back(static_cast<int32_t>(val));
  }
  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(shapeVecI32.size())}, rewriter.getI32Type());
  auto shapeAttr = DenseIntElementsAttr::get(shapeType, shapeVecI32);
  return rewriter.create<arith::ConstantOp>(loc, shapeType, shapeAttr);
}

// Helper function for reshape-transpose-reshape pattern.
LogicalResult
reshapeTransposeReshape(PatternRewriter &rewriter, Value tensor,
                        const SmallVector<int64_t, 4> &reshapeShape,
                        const SmallVector<int64_t, 4> &permVec,
                        const SmallVector<int64_t, 4> &origShape,
                        Value &result) {
  auto loc = tensor.getLoc();
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto elementType = tensorType.getElementType();

  // Reshape tensor to reshapeShapeExclBatch.
  Value newShapeOp = createShapeConstOp(rewriter, loc, reshapeShape);
  if (!newShapeOp)
    return failure();
  auto reshapedType = RankedTensorType::get(reshapeShape, elementType);
  auto reshapedTensor =
      rewriter.create<TFL::ReshapeOp>(loc, reshapedType, tensor, newShapeOp);

  // Convert permVecExclBatch to int32_t vector.
  SmallVector<int32_t, 4> permVecI32;
  for (auto val : permVec) {
    permVecI32.push_back(static_cast<int32_t>(val));
  }

  // Create perm op.
  auto permType = RankedTensorType::get(
      {static_cast<int64_t>(permVecI32.size())}, rewriter.getI32Type());
  auto permAttr = DenseIntElementsAttr::get(permType, permVecI32);
  auto permOp = rewriter.create<arith::ConstantOp>(loc, permType, permAttr);

  // Compute transposed shape.
  SmallVector<int64_t, 4> transposedShape;
  for (auto idx : permVec) {
    if (idx < 0 || idx >= reshapeShape.size())
      return failure();
    transposedShape.push_back(reshapeShape[idx]);
  }
  auto transposedType = RankedTensorType::get(transposedShape, elementType);

  // Transpose.
  auto transposedTensor = rewriter.create<TFL::TransposeOp>(
      loc, transposedType, reshapedTensor, permOp);

  // Reshape back to original shape.
  Value origShapeOp = createShapeConstOp(rewriter, loc, origShape);
  if (!origShapeOp)
    return failure();
  auto finalTensor = rewriter.create<TFL::ReshapeOp>(
      loc, tensorType, transposedTensor, origShapeOp);

  result = finalTensor.getResult();
  return success();
}

} // namespace mlir::xcore::utils
