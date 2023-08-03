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
      sizeInBytes =
          t.getElementType().getIntOrFloatBitWidth() / CHAR_BIT;
    }

    llvm::ArrayRef<int64_t> shape = t.getShape();
    // Handle dynamic shapes
    for (auto &dim : shape) {
      sizeInBytes *= (ShapedType::isDynamic(dim) ? 1 : dim);
    }

    return sizeInBytes;
}

} // namespace utils
} // namespace xcore
} // namespace mlir
