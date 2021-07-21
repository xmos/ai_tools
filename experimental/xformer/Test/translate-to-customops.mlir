// RUN: xcore-opt --mlir-io %s --xcore-translate-to-customop | FileCheck %s

// CHECK-LABEL: valid_xcfc
func @valid_xcfc(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK: tfl.custom
// CHECK-SAME: XC_fc
  %2 = "xc.fc"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: invalid_tflfc
func @invalid_tflfc(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK-NOT: tfl.custom
// CHECK-NOT: XC_fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: activation_lowering
func @activation_lowering(%arg0: tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>> attributes {tf.entry_function = {inputs = "lambda_input_int8", outputs = "Identity_int8"}} {
  %cst = constant dense<"0x828486888A8C8E90929496989A9C9EA0A2A4A6A8AAACAEB0B2B4B6B8BABCBEC0C2C4C6C8CACCCED0D2D4D6D8DADCDEE0E2E4E6E8EAECEEF0F2F4F6F8FAFCFE00020406080A0C0E10121416181A1C1E20222426282A2C2E30323436383A3C3E40424446484A4C4E50525456585A5C5E60626466686A6C6E70727476787A7C7E7F8080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080"> : tensor<256xi8>
// CHECK: tfl.custom
// CHECK-SAME: XC_lookup_8
  %0 = "xc.lookup_8"(%arg0, %cst) : (tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<256xi8>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
  return %0 : tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
}

// -----

// CHECK-LABEL: replace_pad
func @replace_pad(%arg0: tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
    %cst = constant dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>
// CHECK: tfl.custom
// CHECK-SAME: XC_pad
// CHECK-SAME: custom_option = opaque<"xc", "0x7061645F76616C756500010B010101FF04022401">
    %0 = "xc.pad"(%arg0, %cst) {pad_value = -1 : i32} : (tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<4x2xi32>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
    return %0 : tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  }
