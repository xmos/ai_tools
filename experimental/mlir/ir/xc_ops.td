//===----------------------------------------------------------------------===//
//
// This is the operation definition file for XCore dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"

#ifndef TFL_OPS
def TFL_AF_None  : StrEnumAttrCase<"NONE">;
def TFL_AF_Relu  : StrEnumAttrCase<"RELU">;
def TFL_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TFL_AF_Relu6 : StrEnumAttrCase<"RELU6">;

def TFL_AFAttr : StrEnumAttr<
    "ActivationFunctionType", "fused activation enum", [
      TFL_AF_None,  TFL_AF_Relu, TFL_AF_Relu1, TFL_AF_Relu6
    ]>;

def TFL_PAD_Same  : StrEnumAttrCase<"SAME">;
def TFL_PAD_Valid : StrEnumAttrCase<"VALID">;

def TFL_PaddingAttr : StrEnumAttr<"Padding", "padding enum", [
      TFL_PAD_Same, TFL_PAD_Valid
    ]>;

def TFL_FCWO_Default  : StrEnumAttrCase<"DEFAULT">;
def TFL_FCWO_Shuffled4x16i8  : StrEnumAttrCase<"SHUFFLED4x16INT8">;

def TFL_FullyConnectedOptionsWeightFormatAttr :
    StrEnumAttr<"FullyConnectedOptionsWeightsFormat",
                "fully connected options weights format", [
      TFL_FCWO_Default, TFL_FCWO_Shuffled4x16i8
    ]>;
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
class XC_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XCoreDialect, mnemonic, traits> {

  let extraClassDeclaration = [{
    std::vector<uint8_t> buildCustomOptions();
  }];
}

def XC_FullyConnectedOp : XC_Op<"fc", [NoSideEffect]> {
  let summary = "Fully connected op";

  let description = [{
Fully connected op.
  }];

  let arguments = (ins
    TensorOf<[QI8]>:$input,
    TensorOf<[QI8]>:$filter,
    TensorOf<[QI32]>:$bias,

    TFL_AFAttr:$fused_activation_function,
    TFL_FullyConnectedOptionsWeightFormatAttr:$weights_format,
    BoolAttr:$keep_num_dims
  );

  let results = (outs
    TensorOf<[QI8]>:$output
  );
}

#endif // XCORE_DIALECT
