// RUN: xcore-opt --mlir-io %s --xcore-replace-avgpool-with-conv2d | FileCheck %s

// CHECK-LABEL: main2x2
func.func @main2x2(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  
  // CHECK: tfl.depthwise_conv_2d
  // CHECK-SAME: stride_h = 2 : i32
  // CHECK-SAME: stride_w = 2 : i32
  // CHECK-SAME: tensor<1x2x2x4x!quant.uniform<i8:f32, 2.500000e-01>
  // CHECK-NOT: tfl.average_pool_2d
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
  // CHECK: return
  return %0 : tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
}

// CHECK-LABEL: main3x3
func.func @main3x3(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 1.:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  
  // CHECK: tfl.depthwise_conv_2d
  // CHECK-SAME: stride_h = 3 : i32
  // CHECK-SAME: stride_w = 3 : i32
  // CHECK-SAME: tensor<1x3x3x4x!quant.uniform<i8:f32, 0.1111111119389534>
  // CHECK-NOT: tfl.average_pool_2d
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 3 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 1.:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>>
  // CHECK: return
  return %0 : tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>>
}

// CHECK-LABEL: main4x4
func.func @main4x4(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 1.:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  
  // CHECK: tfl.depthwise_conv_2d
  // CHECK-SAME: stride_h = 1 : i32
  // CHECK-SAME: stride_w = 1 : i32
  // CHECK-SAME: tensor<1x4x4x4x!quant.uniform<i8:f32, 6.250000e-02>
  // CHECK-NOT: tfl.average_pool_2d
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 4 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 1.:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>>
  // CHECK: return
  return %0 : tensor<?x1x1x4x!quant.uniform<i8:f32, 1.:124>>
}
