// RUN: xcore-opt --mlir-io %s --xcore-replace-concat | FileCheck %s 

// CHECK-LABEL: valid_concat
func.func @valid_concat(%arg0: tensor<1x2x6x5x2xf32> {tf_saved_model.index_path = ["input_8"]}, %arg1: tensor<1x2x6x5x2xf32> {tf_saved_model.index_path = ["input_7"]}) -> (tensor<1x4x6x5x2xf32> {tf_saved_model.index_path = ["concatenate_3"]}) attributes {tf.entry_function = {inputs = "serving_default_input_8:0,serving_default_input_7:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.concat
  // CHECK-NOT: tfl.concatenation
  %0 = "tfl.quantize"(%arg1) {qtype = tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>} : (tensor<1x2x6x5x2xf32>) -> tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756512105464935>>} : (tensor<1x2x6x5x2xf32>) -> tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756512105464935>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>} : (tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756512105464935>>) -> tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>
  %3 = "tfl.concatenation"(%0, %2) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>, tensor<1x2x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>) -> tensor<1x4x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>
  %4 = "tfl.dequantize"(%3) : (tensor<1x4x6x5x2x!quant.uniform<i16:f32, 0.038756620138883591>>) -> tensor<1x4x6x5x2xf32>
  return %4 : tensor<1x4x6x5x2xf32>
}
