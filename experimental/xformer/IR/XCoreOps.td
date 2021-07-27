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

#ifndef TFL_OPS
def TFL_AF_None : StrEnumAttrCase<"NONE">;
def TFL_AF_Relu : StrEnumAttrCase<"RELU">;
def TFL_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TFL_AF_Relu6 : StrEnumAttrCase<"RELU6">;

def TFL_AFAttr
    : StrEnumAttr<"ActivationFunctionType", "fused activation enum",
                  [TFL_AF_None, TFL_AF_Relu, TFL_AF_Relu1, TFL_AF_Relu6]>;

def TFL_PAD_Same : StrEnumAttrCase<"SAME">;
def TFL_PAD_Valid : StrEnumAttrCase<"VALID">;

def TFL_PaddingAttr
    : StrEnumAttr<"Padding", "padding enum", [TFL_PAD_Same, TFL_PAD_Valid]>;

def TFL_FCWO_Default : StrEnumAttrCase<"DEFAULT">;
def TFL_FCWO_Shuffled4x16i8 : StrEnumAttrCase<"SHUFFLED4x16INT8">;

def TFL_FullyConnectedOptionsWeightFormatAttr
    : StrEnumAttr<"FullyConnectedOptionsWeightsFormat",
                  "fully connected options weights format",
                  [TFL_FCWO_Default, TFL_FCWO_Shuffled4x16i8]>;

// A type attribute containing OpaqueElementsAttr and bytes.
def OpaqueBytesAttr : ElementsAttrBase<
  And<[
    CPred<"$_self.isa<OpaqueElementsAttr>() ">,
    CPred<"$_self.cast<OpaqueElementsAttr>().getType()"
          ".getElementType().isInteger(8)">,
  ]>,
  "opaque bytes attribute"
 > {
  let storageType = [{ OpaqueElementsAttr }];
  let returnType = [{ OpaqueElementsAttr }];
  let convertFromStorage = "$_self";
}
#endif

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
def XC_Conv2D_ValidInDirect : StrEnumAttrCase<"ValidInDirect">;
def XC_Conv2D_PaddedInDirect : StrEnumAttrCase<"PaddedInDirect">;

def XC_Conv2D_TypeAttr
    : StrEnumAttr<"Conv2DType", "conv2d type enum",
                  [XC_Conv2D_ValidDirect, XC_Conv2D_ValidInDirect, XC_Conv2D_PaddedInDirect]>;

def XC_Conv2DV2 : XC_Op<"conv2dv2", [NoSideEffect]> {
  let summary = "Conv2D V2 op";

  let description = [{Conv2D V2 op.}];

  let arguments = (ins
    TensorOf<[QI8, QUI8]>:$input,

    OpaqueBytesAttr:$abstract_kernel_params,
    OpaqueBytesAttr:$memcpy_fn_params,
    OpaqueBytesAttr:$aggregate_fn_params,
    OpaqueBytesAttr:$output_transform_fn_params,
    XC_Conv2D_TypeAttr:$conv2d_type
  );

  let results = (outs TensorOf<[QI8, QUI8]> : $output);
}

def XC_FullyConnectedOp : XC_Op<"fc", [NoSideEffect]> {
  let summary = "Fully connected op";

  let description = [{Fully connected op.}];

  let arguments = (ins
    TensorOf<[QI8]>:$input,
    // TODO: Hack to enable I8 for filter and I16 for bias
    // We might need a different op or match and replace in one pass
    TensorOf<[QI8, I8]>:$filter,
    TensorOf<[QI32, I16]>:$bias,

    TFL_AFAttr:$fused_activation_function,
    TFL_FullyConnectedOptionsWeightFormatAttr:$weights_format,
    BoolAttr:$keep_num_dims
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

#endif // XCORE_DIALECT
