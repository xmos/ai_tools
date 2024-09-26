// RUN: xcore-opt --mlir-io %s --xcore-replace-slice | FileCheck %s 

// CHECK-LABEL: valid_slice1
func.func @valid_slice1(%arg0: tensor<1x8x15x32xi8> {tf_saved_model.index_path = ["input_7"]}) -> (tensor<1x8x15x28xi8> {tf_saved_model.index_path = ["tf.slice_6"]}) attributes {tf.entry_function = {inputs = "serving_default_input_7:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.slice
  // CHECK-NOT: tfl.slice
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 8, 15, 28]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<1x8x15x32xi8>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x8x15x28xi8>
  return %2 : tensor<1x8x15x28xi8>
}
