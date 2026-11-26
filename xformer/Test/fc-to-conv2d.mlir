// RUN: xcore-opt --mlir-io %s --xcore-replace-fc-with-conv2d | FileCheck %s

// CHECK-LABEL: valid_fc
func.func @valid_fc(%arg0: tensor<?x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[-1, 32]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tfl.reshape
  %1 = "tfl.reshape"(%arg0, %0) : (tensor<?x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<2xi32>) -> tensor<?x32x!quant.uniform<i8:f32, 0.0078160231932997704>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<[12671, -12241, 8840, 6018, 4691, -13740, 8148, 4067, -16007, 1746, 11021, -2062, 14847, 8417, 12891, -1799, -5711, -5060, 13417, 9017, 12993, 139, -1615, 3055, 7109, 13545, 15667, -4673, -12692, 2486, -9783, 2882]> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>

  // CHECK: tfl.reshape
  // CHECK-SAME: ?x32
  // CHECK-SAME: 1x1x1x32

  // CHECK: tfl.conv_2d
  // CHECK-SAME: 1x1x1x32
  // CHECK-SAME: 32x1x1x32

  // CHECK: tfl.reshape
  // CHECK-SAME: 1x1x1x32
  // CHECK-SAME: ?x32
  %4 = "tfl.fully_connected"(%1, %2, %3) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x32x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
  return %4 : tensor<?x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// CHECK-LABEL: fc_3d_keep_dims
func.func @fc_3d_keep_dims(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 0.5>>) -> tensor<2x3x5x!quant.uniform<i8:f32, 0.5>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, value = dense<1> : tensor<5x4xi8>} : () -> tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<5x!quant.uniform<i32:f32, 0.25>>, value = dense<0> : tensor<5xi32>} : () -> tensor<5x!quant.uniform<i32:f32, 0.25>>

  // CHECK-DAG: %[[INPUT_SHAPE_3D:.*]] = {{.*}}dense<[1, 1, 6, 4]> : tensor<4xi32>
  // CHECK-DAG: %[[OUTPUT_SHAPE_3D:.*]] = {{.*}}dense<[2, 3, 5]> : tensor<3xi32>
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[INPUT_SHAPE_3D]]
  // CHECK-SAME: 2x3x4x
  // CHECK-SAME: 1x1x6x4x
  // CHECK: tfl.conv_2d
  // CHECK-SAME: 1x1x6x4x
  // CHECK-SAME: 5x1x1x4x
  // CHECK-SAME: 1x1x6x5x
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[OUTPUT_SHAPE_3D]]
  // CHECK-SAME: 1x1x6x5x
  // CHECK-SAME: 2x3x5x
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<2x3x4x!quant.uniform<i8:f32, 0.5>>, tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, tensor<5x!quant.uniform<i32:f32, 0.25>>) -> tensor<2x3x5x!quant.uniform<i8:f32, 0.5>>
  return %2 : tensor<2x3x5x!quant.uniform<i8:f32, 0.5>>
}

// CHECK-LABEL: fc_4d_keep_dims
func.func @fc_4d_keep_dims(%arg0: tensor<2x3x4x4x!quant.uniform<i8:f32, 0.5>>) -> tensor<2x3x4x5x!quant.uniform<i8:f32, 0.5>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, value = dense<1> : tensor<5x4xi8>} : () -> tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<5x!quant.uniform<i32:f32, 0.25>>, value = dense<0> : tensor<5xi32>} : () -> tensor<5x!quant.uniform<i32:f32, 0.25>>

  // CHECK-DAG: %[[INPUT_SHAPE_4D:.*]] = {{.*}}dense<[1, 1, 24, 4]> : tensor<4xi32>
  // CHECK-DAG: %[[OUTPUT_SHAPE_4D:.*]] = {{.*}}dense<[2, 3, 4, 5]> : tensor<4xi32>
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[INPUT_SHAPE_4D]]
  // CHECK-SAME: 2x3x4x4x
  // CHECK-SAME: 1x1x24x4x
  // CHECK: tfl.conv_2d
  // CHECK-SAME: 1x1x24x4x
  // CHECK-SAME: 5x1x1x4x
  // CHECK-SAME: 1x1x24x5x
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[OUTPUT_SHAPE_4D]]
  // CHECK-SAME: 1x1x24x5x
  // CHECK-SAME: 2x3x4x5x
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<2x3x4x4x!quant.uniform<i8:f32, 0.5>>, tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, tensor<5x!quant.uniform<i32:f32, 0.25>>) -> tensor<2x3x4x5x!quant.uniform<i8:f32, 0.5>>
  return %2 : tensor<2x3x4x5x!quant.uniform<i8:f32, 0.5>>
}

// CHECK-LABEL: fc_3d_flatten_output
func.func @fc_3d_flatten_output(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 0.5>>) -> tensor<6x5x!quant.uniform<i8:f32, 0.5>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, value = dense<1> : tensor<5x4xi8>} : () -> tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<5x!quant.uniform<i32:f32, 0.25>>, value = dense<0> : tensor<5xi32>} : () -> tensor<5x!quant.uniform<i32:f32, 0.25>>

  // CHECK-DAG: %[[INPUT_SHAPE_FLAT:.*]] = {{.*}}dense<[1, 1, 6, 4]> : tensor<4xi32>
  // CHECK-DAG: %[[OUTPUT_SHAPE_FLAT:.*]] = {{.*}}dense<[6, 5]> : tensor<2xi32>
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[INPUT_SHAPE_FLAT]]
  // CHECK-SAME: 2x3x4x
  // CHECK-SAME: 1x1x6x4x
  // CHECK: tfl.conv_2d
  // CHECK-SAME: 1x1x6x4x
  // CHECK-SAME: 5x1x1x4x
  // CHECK-SAME: 1x1x6x5x
  // CHECK: tfl.reshape
  // CHECK-SAME: %[[OUTPUT_SHAPE_FLAT]]
  // CHECK-SAME: 1x1x6x5x
  // CHECK-SAME: 6x5x
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3x4x!quant.uniform<i8:f32, 0.5>>, tensor<5x4x!quant.uniform<i8<-127:127>:f32, 0.5>>, tensor<5x!quant.uniform<i32:f32, 0.25>>) -> tensor<6x5x!quant.uniform<i8:f32, 0.5>>
  return %2 : tensor<6x5x!quant.uniform<i8:f32, 0.5>>
}
