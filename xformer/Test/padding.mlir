// RUN: xcore-opt --mlir-io %s --xcore-replace-pad | FileCheck %s

// CHECK-LABEL: replace_pad_negative_zeropoint
func.func @replace_pad_negative_zeropoint(%arg0: tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: xc.pad
  // CHECK-SAME: zero_point = -151587082 : i32
  %1 = "tfl.pad"(%arg0, %0) : (tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>, tensor<4x2xi32>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>
  return %1 : tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>
}

// -----

// CHECK-LABEL: replace_pad_positive_zeropoint
func.func @replace_pad_positive_zeropoint(%arg0: tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: xc.pad
  // CHECK-SAME: zero_point = 168430090 : i32
  %1 = "tfl.pad"(%arg0, %0) : (tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>, tensor<4x2xi32>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
  return %1 : tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
}

// -----

// CHECK-LABEL: replace_pad_with_invalid_shape
func.func @replace_pad_with_invalid_shape(%arg0: tensor<1x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NOT: xc.pad
  %1 = "tfl.pad"(%arg0, %0) : (tensor<1x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<3x2xi32>) -> tensor<1x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %1 : tensor<1x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}

// -----

// CHECK-LABEL: replace_pad_with_batchchannel_padding
func.func @replace_pad_with_batchchannel_padding(%arg0: tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[1, 1], [0, 0], [2, 0], [1, 1]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: xc.pad
  %1 = "tfl.pad"(%arg0, %0) : (tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<4x2xi32>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %1 : tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}
