// RUN: xcore-opt --mlir-io %s --xcore-apply-tflpatterns | FileCheck %s

// CHECK-LABEL: hoist_quantize_above_concat
func @hoist_quantize_above_concat(%arg0: tensor<?x128x128x3x!quant.uniform<i8:f32, 7.812500e-03>> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<?x4032x1x!quant.uniform<i8:f32, 0.017200695350766182:-4>> {tf_saved_model.index_path = ["concatenate_2"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<?x3072x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>, value = dense<1> : tensor<3072x1xi8>} : () -> tensor<?x3072x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<?x768x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>, value = dense<1> : tensor<768x1xi8>} : () -> tensor<?x768x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<?x192x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>, value = dense<1> : tensor<192x1xi8>} : () -> tensor<?x192x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>
  // CHECK: tfl.quantize
  // CHECK: tfl.quantize
  // CHECK: tfl.quantize
  %204 = "tfl.concatenation"(%1, %2, %3) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<?x3072x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>, tensor<?x768x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>, tensor<?x192x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>) -> tensor<?x4032x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>
  // CHECK-NOT: tfl.quantize
  %205 = "tfl.quantize"(%204) {qtype = tensor<?x4032x1x!quant.uniform<i8:f32, 0.017200695350766182:-4>>} : (tensor<?x4032x1x!quant.uniform<i8:f32, 0.11897759139537811:84>>) -> tensor<?x4032x1x!quant.uniform<i8:f32, 0.017200695350766182:-4>>
  return %205 : tensor<?x4032x1x!quant.uniform<i8:f32, 0.017200695350766182:-4>>
}
