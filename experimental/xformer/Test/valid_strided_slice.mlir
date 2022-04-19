// RUN: xcore-opt --mlir-io %s | FileCheck %s 

// CHECK-LABEL: main
func @main(%arg0: tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>> {tf_saved_model.index_path = ["flatten"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {

  // CHECK: XC_strided_slice
  // CHECK-NOT: tfl.strided_slice

  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[160, 80, 4, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tfl.pseudo_const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tfl.strided_slice"(%arg0, %0, %1, %2) {begin_mask = 8 : i32, ellipsis_mask = 0 : i32, end_mask = 8 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %4 = "tfl.pseudo_const"() {value = dense<[0, 80, 0, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
  %5 = "tfl.pseudo_const"() {value = dense<[160, 160, 4, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %6 = "tfl.strided_slice"(%arg0, %4, %5, %2) {begin_mask = 8 : i32, ellipsis_mask = 0 : i32, end_mask = 8 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %7 = "tfl.concatenation"(%3, %6) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>) -> tensor<?x160x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %8 = "tfl.pseudo_const"() {value = dense<[-1, 2560]> : tensor<2xi32>} : () -> tensor<2xi32>
  %9 = "tfl.reshape"(%7, %8) : (tensor<?x160x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<2xi32>) -> tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>>
  // CHECK: return
  return %9 : tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>>
}
