// RUN: xcore-opt --mlir-io %s --xcore-replace-pad | FileCheck %s 

// CHECK-LABEL: valid_pad
func.func @valid_pad(%arg0: tensor<1x8x8x9xf32> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<1x9x9x9xf32> {tf_saved_model.index_path = ["zero_padding2d"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.pad
  // CHECK-NOT: tfl.pad
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %1 = "tfl.pad"(%arg0, %0) : (tensor<1x8x8x9xf32>, tensor<4x2xi32>) -> tensor<1x9x9x9xf32>
  return %1 : tensor<1x9x9x9xf32>
}
