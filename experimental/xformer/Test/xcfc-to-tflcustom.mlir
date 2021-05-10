// RUN: xcore-opt --mlir-io %s --xcore-translate-to-customop | FileCheck %s

// CHECK-LABEL: valid_xcfc
func @valid_xcfc(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK: tfl.custom
// CHECK-SAME: XC_fc
// CHECK-SAME: 0x696C6C6567616C5F706172616D730001100101010104022401
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
// CHECK-NOT: 0x696C6C6567616C5F706172616D730001100101010104022401
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}
