// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

//===----------------------------------------------------------------------===//
//
// This is the operation definition file for XCore dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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
class XC_Op<string mnemonic, list<Trait> traits = []>
    : Op<XCoreDialect, mnemonic, traits> {

  let extraClassDeclaration = [{ std::vector<uint8_t> buildCustomOptions(); }];
}

// Conv2D

def XC_Conv2D_ValidDirect : I32EnumAttrCase<"ValidDirect", 0>;
def XC_Conv2D_ValidIndirect : I32EnumAttrCase<"ValidIndirect", 1>;
def XC_Conv2D_PaddedIndirect : I32EnumAttrCase<"PaddedIndirect", 2>;
def XC_DW_Conv2D_ValidDirect : I32EnumAttrCase<"DepthwiseValidDirect", 3>;
def XC_DW_Conv2D_PaddedIndirect : I32EnumAttrCase<"DepthwisePaddedIndirect", 4>;
def XC_BNN_Conv2D_ValidDirect_Binary
    : I32EnumAttrCase<"BNNValidDirectBinary", 5>;
def XC_BNN_Conv2D_ValidIndirect_Binary
    : I32EnumAttrCase<"BNNValidIndirectBinary", 6>;
def XC_BNN_Conv2D_ValidDirect_Int8 : I32EnumAttrCase<"BNNValidDirectInt8", 7>;
def XC_BNN_Conv2D_ValidIndirect_Int8
    : I32EnumAttrCase<"BNNValidIndirectInt8", 8>;
def XC_Conv2D_ValidDirect_Int16 : I32EnumAttrCase<"ValidDirectI16", 9>;

def XC_Conv2D_TypeAttr : I32EnumAttr<"Conv2DType", "conv2d type enum", [
  XC_Conv2D_ValidDirect, XC_Conv2D_ValidIndirect, XC_Conv2D_PaddedIndirect,
  XC_DW_Conv2D_ValidDirect, XC_DW_Conv2D_PaddedIndirect,
  XC_BNN_Conv2D_ValidDirect_Binary, XC_BNN_Conv2D_ValidIndirect_Binary,
  XC_BNN_Conv2D_ValidDirect_Int8, XC_BNN_Conv2D_ValidIndirect_Int8,
  XC_Conv2D_ValidDirect_Int16
]>;

def XC_Conv2D_Group : I32EnumAttrCase<"Group", 0>;
def XC_Conv2D_Channelwise : I32EnumAttrCase<"Channelwise", 1>;

def XC_Conv2D_OTAttr : I32EnumAttr<"OtType", "output transform type enum",
                                   [XC_Conv2D_Group, XC_Conv2D_Channelwise]>;

def XC_SliceOp : XC_Op<"slice", [Pure]> {
  let summary = "Slice op";

  let description = [{Slice op.}];

  let arguments = (ins
    TensorOf<[QI8, QI16, F32, I8, I32]>:$input,

    I32Attr:$start,
    I32Attr:$offset,
    I32Attr:$size,
    I32Attr:$num_copies,
    BoolAttr:$use_vpu
  );

  let results = (outs TensorOf<[QI8, QI16, F32, I8, I32]> : $output);
}

def XC_PadOp : XC_Op<"pad", [Pure]> {
  let summary = "Pad op";

  let description = [{Pad op.}];

  let arguments = (ins
    TensorOf<[QI8, QI16, F32, I8, I32]>:$input,

    I32Attr:$start,
    I32Attr:$pad_size,
    I32Attr:$size,
    I32Attr:$num_copies,
    I32Attr:$zero_point,
    I32Attr:$end,
    BoolAttr:$use_vpu
  );

  let results = (outs TensorOf<[QI8, QI16, F32, I8, I32]> : $output);

  let hasFolder = 1;
}

def XC_ConcatOp : XC_Op<"concat", [Pure]> {
  let summary = "Concat op";

  let description = [{Concat op.}];

  let arguments = (ins
    Variadic<TensorOf<[QI8, QI16, F32, I8, I32]>> : $input,

    I32Attr:$num_copies,
    I32ArrayAttr:$sizes,
    I32Attr:$num_inputs,
    BoolAttr:$use_vpu
  );

  let results = (outs TensorOf<[QI8, QI16, F32, I8, I32]> : $output);
}

def XC_AddOp : XC_Op<"add", [Pure]> {
  let summary = "Add op";

  let description = [{Add op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input1,
    TensorOf<[QI8]>:$input2,
    StrAttr:$fused_activation_function,

    I32Attr:$multiplier1,
    I32Attr:$multiplier2,
    I32Attr:$bias,
    I32Attr:$shift
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_MulOp : XC_Op<"mul", [Pure]> {
  let summary = "Mul op";

  let description = [{Mul op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input1,
    TensorOf<[QI8]>:$input2,

    StrAttr:$mul_params
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_UnaryI16_Quantize : I32EnumAttrCase<"Quantize", 0>;
def XC_UnaryI16_Requantize : I32EnumAttrCase<"Requantize", 1>;
def XC_UnaryI16_Dequantize : I32EnumAttrCase<"Dequantize", 2>;
def XC_UnaryI16_OpTypeAttr : I32EnumAttr<"UnaryI16OpType", "op type enum", [
  XC_UnaryI16_Quantize, XC_UnaryI16_Requantize, XC_UnaryI16_Dequantize
]>;

def XC_UnaryI16Op : XC_Op<"unaryi16", [Pure]> {
  let summary = "Unary I16 op";

  let description = [{Unary I16 op.}];

  let arguments = (ins
    TensorOf<[QI16, F32]> : $input,
    TensorOf<[UI8]> : $blob,

    I32Attr:$op_type
    );

  let results = (outs TensorOf<[QI16, F32]> : $output);
}

def XC_BinaryI16_Add : I32EnumAttrCase<"Add", 0>;
def XC_BinaryI16_Mul : I32EnumAttrCase<"Mul", 1>;
def XC_BinaryI16_OpTypeAttr : I32EnumAttr<"BinaryI16OpType", "op type enum",
                                          [XC_BinaryI16_Add, XC_BinaryI16_Mul]>;

def XC_BinaryI16Op : XC_Op<"binaryi16", [Pure]> {
  let summary = "Binary I16 op";

  let description = [{Binary I16 op.}];

  let arguments = (ins
    TensorOf<[QI16]> : $input1,
    TensorOf<[QI16]> : $input2,
    TensorOf<[UI8]> : $blob,

    I32Attr:$op_type
    );

  let results = (outs TensorOf<[QI16]> : $output);
}

def XC_Beta_ActivationF32Op : XC_Op<"beta_activationf32", [Pure]> {
  let summary = "Beta ActivationF32 op";

  let description = [{Beta ActivationF32 op.}];

  let arguments = (ins TensorOf<[F32]> : $input, I32Attr : $type);

  let results = (outs TensorOf<[F32]> : $output);
}

def XC_Beta_ConcatF32Op : XC_Op<"beta_concatf32", [Pure]> {
  let summary = "Beta ConcatF32 op";

  let description = [{Beta ConcatF32 op.}];

  let arguments = (ins Variadic<TensorOf<[F32]>> : $input);

  let results = (outs TensorOf<[F32]> : $output);
}

def XC_Beta_ConvF32Op : XC_Op<"beta_convf32", [Pure]> {
  let summary = "Beta ConvF32 op";

  let description = [{Beta ConvF32 op.}];

  let arguments = (ins
    TensorOf<[F32]>:$input,
    TensorOf<[F32, UI8]>:$kernels,
    TensorOf<[F32]>:$bias
  );

  let results = (outs TensorOf<[F32]> : $output);
}

def XC_Beta_TransposeConvF32Op : XC_Op<"beta_transposeconvf32", [Pure]> {
  let summary = "Beta Transpose ConvF32 op";

  let description = [{Beta Transpose ConvF32 op.}];

  let arguments = (ins
    TensorOf<[F32]>:$input,
    TensorOf<[F32, UI8]>:$kernels,
    TensorOf<[F32]>:$bias
  );

  let results = (outs TensorOf<[F32]> : $output);
}

def XC_Beta_FcF32Op : XC_Op<"beta_fcf32", [Pure]> {
  let summary = "Beta Fc F32 op";

  let description = [{Beta Fc ConvF32 op.}];

  let arguments = (ins
    TensorOf<[F32]>:$input,
    TensorOf<[F32, UI8]>:$kernels
  );

  let results = (outs TensorOf<[F32]> : $output);
}

def XC_MaxPool2DOp : XC_Op<"maxpool2d", [Pure]> {
  let summary = "MaxPool2D op";

  let description = [{MaxPool2D op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input,
    // The scratch buffer tensor
    AnyTypeOf<[TensorOf<[I8]>, NoneType]>:$scratch_tensor,
    StrAttr:$memcpy_fn_param,
    StrAttr:$aggregate_fn_param,
    StrAttr:$output_transform_fn_param,
    I32Attr:$scratch_bytes,
    I32Attr:$thread_count,
    StrArrayAttr:$abstract_kernel_params
  );

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_SoftmaxOp : XC_Op<"softmax", [Pure]> {
  let summary = "Softmax op";

  let description = [{Softmax op.}];

  let arguments = (ins TensorOf<[QI8]> : $input, TensorOf<[F32]> : $lut);

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_Conv2DV2Op : XC_Op<"conv2d_v2", [Pure]> {
  let summary = "Conv2D V2 op";

  let description = [{Conv2D V2 op.}];

  let arguments = (ins
    // I32 input is for BNNs
    TensorOf<[QI8, QI16, I32]>:$input,
    TensorOf<[I8]>:$weights,
    TensorOf<[I16, I32]>:$mulsbiases_or_thresholds,
    // The partial output parameter is used when we chain convolutions
    AnyTypeOf<[TensorOf<[QI8, QI16]>, NoneType]>:$partial_output,
    // The scratch buffer tensor
    AnyTypeOf<[TensorOf<[I8]>, NoneType]>:$scratch_tensor,
    StrAttr:$conv2d_kernel_type,
    StrAttr:$memcpy_fn_param,
    StrAttr:$aggregate_fn_param,
    StrAttr:$output_transform_fn_param,
    StrAttr:$output_transform_type,
    I32Attr:$scratch_bytes,
    I32Attr:$thread_count,
    StrArrayAttr:$abstract_kernel_params
  );

  // I32 output is for BNNs
  let results = (outs TensorOf<[QI8, QI16, I32]> : $output);
}

// This op is not exported to the flatbuffer
def XC_FakeScratchBufferOp : XC_Op<"fake_scratch", [Pure]> {
  let summary = "Fake scratch buffer op";

  let description = [{Fake scratch buffer op.}];

  let results = (outs TensorOf<[I8]> : $output);
}

def XC_FakeConv2DOp : XC_Op<"fake_conv2d", [Pure]> {
  let summary = "Fake Conv2D op";

  let description = [{Fake Conv2D op.}];

  let arguments = (ins
    TensorOf<[QI8, QI16]>:$input,
    TensorOf<[QI8]>:$filter,
    AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$bias,
    // The partial output parameter is used when we chain convolutions
    AnyTypeOf<[TensorOf<[QI8, QI16]>, NoneType]>:$partial_output,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    StrAttr:$fused_activation_function,
    StrAttr:$padding,
    // For using explicit padding
    AnyTypeOf<[TensorOf<[I32,I64]>, NoneType]>:$padding_values,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    // The following parameters are used when we chain convolutions
    // so that we can read and write to the correct locations
    I32Attr:$output_sub_h,
    I32Attr:$output_sub_w,
    I32Attr:$output_stride_h,
    I32Attr:$output_stride_w,
    I32Attr:$input_offset
  );

  let results = (outs TensorOf<[QI8, QI16]> : $output);
}

def XC_FakeDepthwiseConv2DOp : XC_Op<"fake_depthwise_conv2d", [Pure]> {
  let summary = "Fake Depthwise Conv2D op";

  let description = [{Fake Depthwise Conv2D op.}];

  let arguments = (ins
    TensorOf<[QI8, QI16]>:$input,
    TensorOf<[QI8]>:$filter,
    AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$bias,
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

  let results = (outs TensorOf<[QI8, QI16]> : $output);
}

def XC_LookupOp : XC_Op<"lookup", [Pure]> {
  let summary = "Lookup table op";

  let description = [{Lookup table op.}];

  let arguments = (ins TensorOf<[QI8, QI16]> : $input, TensorOf<[UI8]> : $lut);

  let results = (outs TensorOf<[QI8, QI16]> : $output);
}

def XC_Pad3To4Op : XC_Op<"pad_3_to_4", [Pure]> {
  let summary = "Pad 3 to 4 op";

  let description = [{Pad 3 to 4 op.}];

  let arguments = (ins TensorOf<[QI8]> : $input, I32Attr : $pad_value);

  let results = (outs TensorOf<[QI8]> : $output);
}

def XC_LoadConstantOp
    : XC_Op<"ld_constant", [Pure, SameOperandsAndResultType]> {
  let summary = "Load constant op";

  let description = [{Load constant op.}];

  let arguments = (ins AnyTensor : $input);

  let results = (outs AnyTensor : $output);
}

def XC_LoadFlashOp : XC_Op<"ld_flash", [Pure]> {
  let summary = "Load from flash op";

  let description = [{Load from flash op.}];

  let arguments = (ins I32Attr : $address, I32ArrayAttr : $sizes);

  let results = (outs Variadic<AnyTensor> : $output);
}

def XC_Bsign8Op : XC_Op<"bsign_8", [Pure]> {
  let summary = "Binary sign op";

  let description = [{Binary sign op.}];

  let arguments = (ins TensorOf<[QI8]> : $input);

  let results = (outs TensorOf<[I32]> : $output);
}

#endif // XCORE_DIALECT
