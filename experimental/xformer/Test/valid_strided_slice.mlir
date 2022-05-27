// RUN: xcore-opt --mlir-io %s --xcore-replace-stridedslice | FileCheck %s 

// CHECK-LABEL: main
func @main(%arg0: tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>> {tf_saved_model.index_path = ["flatten"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {

  // CHECK: xc.strided_slice
  // CHECK: xc.strided_slice
  // CHECK-NOT: tfl.strided_slice

  %cst = arith.constant dense<0> : tensor<4xi32>
  %cst_0 = arith.constant dense<[160, 80, 4, 4]> : tensor<4xi32>
  %cst_1 = arith.constant dense<1> : tensor<4xi32>
  %cst_2 = arith.constant dense<[0, 80, 0, 0]> : tensor<4xi32>
  %cst_3 = arith.constant dense<[160, 160, 4, 4]> : tensor<4xi32>
  %cst_4 = arith.constant dense<[-1, 2560]> : tensor<2xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 8 : i32, ellipsis_mask = 0 : i32, end_mask = 8 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %1 = "tfl.strided_slice"(%arg0, %cst_2, %cst_3, %cst_1) {begin_mask = 8 : i32, ellipsis_mask = 0 : i32, end_mask = 8 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<?x160x160x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %2 = "tfl.concatenation"(%0, %1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<?x80x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>) -> tensor<?x160x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>
  %3 = "tfl.reshape"(%2, %cst_4) : (tensor<?x160x4x4x!quant.uniform<i8:f32, 0.007841695100069046>>, tensor<2xi32>) -> tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>>
  // CHECK: return
  return %3 : tensor<?x2560x!quant.uniform<i8:f32, 0.007841695100069046>>
}
