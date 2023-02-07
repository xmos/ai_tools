// RUN: xcore-opt --mlir-io %s --xcore-translate-to-customop | FileCheck %s

// CHECK-LABEL: activation_lowering
func.func @activation_lowering(%arg0: tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>> attributes {tf.entry_function = {inputs = "lambda_input_int8", outputs = "Identity_int8"}} {
  %cst = arith.constant dense<"0x828486888A8C8E90929496989A9C9EA0A2A4A6A8AAACAEB0B2B4B6B8BABCBEC0C2C4C6C8CACCCED0D2D4D6D8DADCDEE0E2E4E6E8EAECEEF0F2F4F6F8FAFCFE00020406080A0C0E10121416181A1C1E20222426282A2C2E30323436383A3C3E40424446484A4C4E50525456585A5C5E60626466686A6C6E70727476787A7C7E7F8080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080"> : tensor<256xi8>
// CHECK: tfl.custom
// CHECK-SAME: XC_lookup
  %0 = "xc.lookup"(%arg0, %cst) : (tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<256xi8>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
  return %0 : tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
}

// -----

// CHECK-LABEL: replace_pad
func.func @replace_pad(%arg0: tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
// CHECK: tfl.custom
// CHECK-SAME: XC_pad
// CHECK-SAME: custom_option = opaque<"xc", "0x7070001800000000040000006000000030000000000000000000000000707600022105000300000001000000020000002C0000000A0A0A0A14060A2601">
    %0 = "xc.pad"(%arg0) {pad_value = 168430090 : i32, padding_plan = "\00\00\00\00\04\00\00\00`\00\00\000\00\00\00\00\00\00\00\00\00\00\00"} : (tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
    return %0 : tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
  }
