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

template <typename ConcreteType>
class NonModifying : public TraitBase<ConcreteType, NonModifying> {};

template <typename ConcreteType>
class OnlyOverlappableWithInput
    : public TraitBase<ConcreteType, OnlyOverlappableWithInput> {};

} // namespace xcore
} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "IR/XCoreOps.h.inc"

constexpr int CONCAT_OP_MAX_INPUTS = 13;

constexpr char kMetadataXCOffsets[] = "xc.offsets";
constexpr char kMetadataXCPeakOpId[] = "xc.peak_op_id";
constexpr char kMetadataXCPeakUsage[] = "xc.peak_usage";
constexpr char kMetadataXCNumExternalInputTensors[] =
    "xc.num_external_input_tensors";
constexpr char kMetadataXCNumExternalOutputTensors[] =
    "xc.num_external_output_tensors";
constexpr char kMetadataXCNumExternalInputTensorsData[] =
    "xc.num_external_input_tensors_data";
constexpr char kMetadataXCNumExternalOutputTensorsData[] =
    "xc.num_external_output_tensors_data";

#endif // XFORMER_IR_XCOREOPS_H
