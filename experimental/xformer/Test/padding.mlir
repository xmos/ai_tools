// RUN: xcore-opt --mlir-io %s --xcore-apply-patterns | FileCheck %s

// CHECK-LABEL: relu
func @relu(%arg0: tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>> attributes {tf.entry_function = {inputs = "lambda_input_int8", outputs = "Identity_int8"}} {
  // CHECK: "0x828486888A8C8E90929496989A9C9EA0A2A4A6A8AAACAEB0B2B4B6B8BABCBEC0C2C4C6C8CACCCED0D2D4D6D8DADCDEE0E2E4E6E8EAECEEF0F2F4F6F8FAFCFE00020406080A0C0E10121416181A1C1E20222426282A2C2E30323436383A3C3E40424446484A4C4E50525456585A5C5E60626466686A6C6E70727476787A7C7E7F8080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080"
  // CHECK-SAME: tensor<256xi8>
  // CHECK: xc.lookup_8
  %0 = "tfl.relu"(%arg0) : (tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
  return %0 : tensor<1x12x4x7x!quant.uniform<i8:f32, 0.0039192917756736279:-128>>
}

// -----

// CHECK-LABEL: valid_padding_negative_zeropoint
func @valid_padding_negative_zeropoint(%arg0: tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: xc.pad
  // CHECK-SAME: {pad_value = -151587082 : i32}
  %1 = "tfl.pad"(%arg0, %0) : (tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>, tensor<4x2xi32>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>
  return %1 : tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-10>>
}

// -----

// CHECK-LABEL: valid_padding_positive_zeropoint
func @valid_padding_positive_zeropoint(%arg0: tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: xc.pad
  // CHECK-SAME: {pad_value = 168430090 : i32}
  %1 = "tfl.pad"(%arg0, %0) : (tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>, tensor<4x2xi32>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
  return %1 : tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:10>>
}

// -----

// CHECK-LABEL: padding_with_invalid_shape
func @padding_with_invalid_shape(%arg0: tensor<?x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<?x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NOT: xc.pad
  %1 = "tfl.pad"(%arg0, %0) : (tensor<?x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<3x2xi32>) -> tensor<?x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %1 : tensor<?x4x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}

// -----

// CHECK-LABEL: padding_with_batchchannel_padding
func @padding_with_batchchannel_padding(%arg0: tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[1, 1], [0, 0], [2, 0], [1, 1]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK-NOT: xc.pad
  %1 = "tfl.pad"(%arg0, %0) : (tensor<?x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<4x2xi32>) -> tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %1 : tensor<?x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}

// -----

// CHECK-LABEL: padding_with_invalid_bytes_per_pixel
func @padding_with_invalid_bytes_per_pixel(%arg0: tensor<?x4x1x47x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<?x4x3x47x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> attributes {tf.entry_function = {inputs = "zero_padding2d_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK-NOT: xc.pad
  %1 = "tfl.pad"(%arg0, %0) : (tensor<?x4x1x47x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>, tensor<4x2xi32>) -> tensor<?x4x3x47x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %1 : tensor<?x4x3x47x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}
