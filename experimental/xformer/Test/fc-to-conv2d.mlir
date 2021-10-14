// RUN: xcore-opt --mlir-io %s --xcore-replace-fc-with-conv2d | FileCheck %s

// CHECK-LABEL: valid_fc
func @valid_fc(%arg0: tensor<?x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[-1, 32]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tfl.reshape
  %1 = "tfl.reshape"(%arg0, %0) : (tensor<?x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<2xi32>) -> tensor<?x32x!quant.uniform<i8:f32, 0.0078160231932997704>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<[12671, -12241, 8840, 6018, 4691, -13740, 8148, 4067, -16007, 1746, 11021, -2062, 14847, 8417, 12891, -1799, -5711, -5060, 13417, 9017, 12993, 139, -1615, 3055, 7109, 13545, 15667, -4673, -12692, 2486, -9783, 2882]> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

  // CHECK: tfl.reshape
  // CHECK-SAME: ?x32
  // CHECK-SAME: ?x1x1x32

  // CHECK: tfl.conv_2d
  // CHECK-SAME: ?x1x1x32
  // CHECK-SAME: 32x1x1x32

  // CHECK: tfl.reshape
  // CHECK-SAME: ?x1x1x32
  // CHECK-SAME: ?x32
  %4 = "tfl.fully_connected"(%1, %2, %3) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x32x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
  return %4 : tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}