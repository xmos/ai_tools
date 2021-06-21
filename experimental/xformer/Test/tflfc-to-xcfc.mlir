// RUN: xcore-opt --mlir-io %s --xcore-apply-patterns | FileCheck %s

// CHECK-LABEL: valid_tflfc
func @valid_tflfc(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK: xc.fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: tflfc_with_invalid_input
func @tflfc_with_invalid_input(%arg0: tensor<1x4x8x1xf32>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK-NOT: xc.fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1xf32>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: tflfc_with_invalid_filter
func @tflfc_with_invalid_filter(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32xf32>, value = dense<1.0> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

// CHECK-NOT: xc.fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32xf32>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: tflfc_with_invalid_bias
func @tflfc_with_invalid_bias(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32xf32>, value = dense<1.0> : tensor<32xf32>} : () -> tensor<32xf32>

// CHECK-NOT: xc.fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32xf32>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}
