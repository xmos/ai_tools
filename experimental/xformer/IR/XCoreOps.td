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

def XC_StridedSliceOp : XC_Op<"strided_slice", [NoSideEffect]> {
  let summary = "Strided Slice op";

  let description = [{Strided Slice op.}];

   let arguments = (ins
    TensorOf<[QI8]>:$input,
    TensorOf<[I32]>:$begin,
    TensorOf<[I32]>:$end,
    TensorOf<[I32]>:$strides
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_StridedSliceV3Op : XC_Op<"strided_slice_v3", [NoSideEffect]> {
  let summary = "Strided Slice V3 op";

  let description = [{Strided Slice V3 op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input,

    I32Attr:$begin_x,
    I32Attr:$begin_y,
    StrAttr:$memcpy_fn_param
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_Conv2DV2Op : XC_Op<"conv2d_v2", [NoSideEffect]> {
  let summary = "Conv2D V2 op";

  let description = [{Conv2D V2 op.}];

  let arguments = (ins
    // I32 input is for BNNs
    TensorOf<[QI8, I32]>:$input,

    TensorOf<[I8]>:$weights,
    TensorOf<[I16]>:$mulsbiases_or_thresholds,
    StrAttr:$conv2d_kernel_type,
    StrAttr:$memcpy_fn_param,
    StrAttr:$aggregate_fn_param,
    StrAttr:$output_transform_fn_param,
    I32Attr:$scratch_bytes,
    I32Attr:$thread_count,
    StrArrayAttr:$abstract_kernel_params
  );

  // I32 output is for BNNs
  let results = (outs TensorOf<[QI8, I32]> : $output);
}

def XC_FakeConv2DOp : XC_Op<"fake_conv2d", [NoSideEffect]> {
  let summary = "Fake Conv2D op";

  let description = [{Fake Conv2D op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input,
    TensorOf<[QI8]>:$filter,
    TensorOf<[I32,QI32]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    StrAttr:$fused_activation_function,
    StrAttr:$padding,
    // For using explicit padding
    AnyTypeOf<[TensorOf<[I32,I64]>, NoneType]>:$padding_values,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_FakeDepthwiseConv2DOp : XC_Op<"fake_depthwise_conv2d", [NoSideEffect]> {
  let summary = "Fake Depthwise Conv2D op";

  let description = [{Fake Depthwise Conv2D op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input,
    TensorOf<[QI8]>:$filter,
    TensorOf<[I32,QI32]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    StrAttr:$fused_activation_function,
    StrAttr:$padding,
    // For using explicit padding
    AnyTypeOf<[TensorOf<[I32,I64]>, NoneType]>:$padding_values,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    I32Attr:$depth_multiplier
  );

  let results = (outs TensorOf<[QI8]> : $output);
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

def XC_LoadConstantOp
    : XC_Op<"ld_constant", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Load constant op";

  let description = [{Load constant op.}];

  let arguments = (ins AnyTensor : $input);

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
