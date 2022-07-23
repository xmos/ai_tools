// RUN: xcore-opt --mlir-io %s --xcore-replace-conv2d --xcore-conv-err-threshold=0.0 | FileCheck %s

// RUN: xcore-opt --mlir-io %s --xcore-replace-conv2d --xcore-conv-err-threshold=1.0 | FileCheck %s -check-prefix=REPLACE-CHECK

// RUN: xcore-opt --mlir-io %s --xcore-replace-conv2d --xcore-conv-err-threshold=0.25 --xcore-force-conv-err-full-check | FileCheck %s -check-prefix=REPLACE2-CHECK

// CHECK-LABEL: notreplaced
func @notreplaced(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, value = dense<1> : tensor<1x2x2x4xi8>} : () -> tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>
  %1 = "tfl.depthwise_conv_2d"(%arg0, %0, %cst) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>, tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<4xi32>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
  // CHECK: tfl.depthwise_conv_2d
  return %1 : tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
}

// -----

// REPLACE-CHECK-LABEL: replaced1
func @replaced1(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, value = dense<1> : tensor<1x2x2x4xi8>} : () -> tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>
  %1 = "tfl.depthwise_conv_2d"(%arg0, %0, %cst) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>, tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<4xi32>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
  // REPLACE-CHECK: xc.conv2d_v2
  // REPLACE-CHECK-NOT: tfl.depthwise_conv_2d
  return %1 : tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
}

// -----

// REPLACE2-CHECK-LABEL: replaced2
func @replaced2(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, value = dense<1> : tensor<1x2x2x4xi8>} : () -> tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>
  %1 = "tfl.depthwise_conv_2d"(%arg0, %0, %cst) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>, tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<4xi32>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
  // REPLACE2-CHECK: xc.conv2d_v2
  // REPLACE2-CHECK-NOT: tfl.depthwise_conv_2d
  return %1 : tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
}
