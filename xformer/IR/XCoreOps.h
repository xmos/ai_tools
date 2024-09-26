// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_IR_XCOREOPS_H
#define XFORMER_IR_XCOREOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "IR/XCoreDialect.h.inc"
// clang-format on

#include "IR/XCoreEnumOps.h.inc"

namespace mlir {
namespace OpTrait {
namespace xcore {

template <typename ConcreteType>
class MemoryOverlappable : public TraitBase<ConcreteType, MemoryOverlappable> {
};

} // namespace xcore
} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "IR/XCoreOps.h.inc"

constexpr int CONCAT_OP_MAX_INPUTS = 13;

#endif // XFORMER_IR_XCOREOPS_H
