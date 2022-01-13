// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

//===----------------------------------------------------------------------===//
//
// This is the operation definition file for XCore dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"

//===----------------------------------------------------------------------===//
// XCore dialect definitions
//===----------------------------------------------------------------------===//

#ifndef XCORE_DIALECT
#define XCORE_DIALECT

def XCoreDialect : Dialect {
  let name = "xc";

  let summary = "Types and operations for XCore dialect";
  let description = [{
    This dialect contains operations for XCore. This dialect will be used in
    conjunction with the TensorFlow dialects for converting & optimizing
    TF graphs to be deployed on XCore.
  }];

  let cppNamespace = "::mlir::xcore";
}

//===----------------------------------------------------------------------===//
// XCore op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class XC_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<XCoreDialect, mnemonic, traits> {

  let extraClassDeclaration = [{ std::vector<uint8_t> buildCustomOptions(); }];
}

// Conv2D

def XC_Conv2D_ValidDirect : StrEnumAttrCase<"ValidDirect">;
def XC_Conv2D_ValidIndirect : StrEnumAttrCase<"ValidIndirect">;
def XC_Conv2D_PaddedIndirect : StrEnumAttrCase<"PaddedIndirect">;
def XC_DW_Conv2D_ValidDirect : StrEnumAttrCase<"DepthwiseValidDirect">;
def XC_DW_Conv2D_PaddedIndirect : StrEnumAttrCase<"DepthwisePaddedIndirect">;
def XC_BNN_Conv2D_ValidDirect_Binary : StrEnumAttrCase<"BNNValidDirectBinary">;
def XC_BNN_Conv2D_ValidIndirect_Binary
    : StrEnumAttrCase<"BNNValidIndirectBinary">;
def XC_BNN_Conv2D_ValidDirect_Int8 : StrEnumAttrCase<"BNNValidDirectInt8">;
def XC_BNN_Conv2D_ValidIndirect_Int8 : StrEnumAttrCase<"BNNValidIndirectInt8">;

def XC_Conv2D_TypeAttr : StrEnumAttr<"Conv2DType", "conv2d type enum", [
  XC_Conv2D_ValidDirect, XC_Conv2D_ValidIndirect, XC_Conv2D_PaddedIndirect,
  XC_DW_Conv2D_ValidDirect, XC_DW_Conv2D_PaddedIndirect,
  XC_BNN_Conv2D_ValidDirect_Binary, XC_BNN_Conv2D_ValidIndirect_Binary,
  XC_BNN_Conv2D_ValidDirect_Int8, XC_BNN_Conv2D_ValidIndirect_Int8
]>;

def XC_Conv2DV2Op : XC_Op<"conv2d_v2", [NoSideEffect]> {
  let summary = "Conv2D V2 op";

  let description = [{Conv2D V2 op.}];

  let arguments = (ins
    // I32 input is for BNNs
    TensorOf<[QI8, I32]>:$input,

    I32Attr:$thread_count,
    I32ArrayAttr:$scratch_bytes,
    TensorOf<[I8]>:$weights,
    TensorOf<[I16]>:$mulsbiases_or_thresholds,
    StrArrayAttr:$abstract_kernel_params,
    StrArrayAttr:$memcpy_fn_params,
    StrArrayAttr:$aggregate_fn_params,
    StrArrayAttr:$output_transform_fn_params,
    StrArrayAttr:$conv2d_kernel_type
  );

  // I32 output is for BNNs
  let results = (outs TensorOf<[QI8, I32]> : $output);
}

def XC_Lookup8Op : XC_Op<"lookup_8", [NoSideEffect]> {
  let summary = "Lookup table op";

  let description = [{Lookup table op.}];

  let arguments = (ins TensorOf<[QI8]> : $input, TensorOf<[I8]> : $lut);

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_PadOp : XC_Op<"pad", [NoSideEffect]> {
  let summary = "Pad op";

  let description = [{Pad op.}];

  let arguments = (ins
    TensorOf<[F32, I32, QI8]>:$input,
    TensorOf<[I32]>:$padding,

    I32Attr:$pad_value
    );

  let results = (outs TensorOf<[F32, I32, QI8]> : $output);
}

def XC_LoadConstantOp : XC_Op<"ld_constant", [NoSideEffect]> {
  let summary = "Load constant op";

  let description = [{Load constant op.}];

  let arguments = (ins ElementsAttr : $input);

  let results = (outs AnyTensor : $output);
}

def XC_LoadFlashOp : XC_Op<"ld_flash", [NoSideEffect]> {
  let summary = "Load from flash op";

  let description = [{Load from flash op.}];

  let arguments = (ins I32Attr : $address, I32Attr : $size);

  let results = (outs AnyTensor : $output);
}

def XC_Bsign8Op : XC_Op<"bsign_8", [NoSideEffect]> {
  let summary = "Binary sign op";

  let description = [{Binary sign op.}];

  let arguments = (ins TensorOf<[QI8]> : $input);

  let results = (outs TensorOf<[I32]> : $output);
}

#endif // XCORE_DIALECT
