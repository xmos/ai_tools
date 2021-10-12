// RUN: xcore-opt --mlir-io %s --xcore-replace-avgpool-with-conv2d | FileCheck %s

// CHECK-LABEL: some_test_name
func @main(%arg0: tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>> attributes {tf.entry_function = {inputs = "average_pooling2d_input", outputs = "Identity"}} {
  
  // CHECK: tfl.depthwise_conv_2d
  // CHECK-SAME: {stride_h = 2 : i32}
  // CHECK-SAME: {stride_w = 2 : i32}
  // CHECK-SAME: {filter_width = 2 : i32}
  // CHECK-SAME: {filter_width = 2 : i32}
  // CHECK-SAME: {depth_multiplier = 1 : i32}
  // CHECK-SAME: {dilation_h_factor = 1 : i32}
  // CHECK-SAME: {dilation_w_factor = 1 : i32}
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x3x3x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
  // CHECK: return
  return %0 : tensor<?x1x1x4x!quant.uniform<i8:f32, 3.9605339406989515E-4:124>>
}