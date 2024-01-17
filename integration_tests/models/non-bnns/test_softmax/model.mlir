// -----// IR Dump After mlir::TFL::TranslateToLCE (lce-translate-tfl) //----- //
func.func @main(%arg0: tensor<?x1000x!quant.uniform<i8:f32, 0.039215190336108208:-128>> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>> {tf_saved_model.index_path = ["softmax"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<?x1000x!quant.uniform<i8:f32, 0.039215190336108208:-128>>) -> tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}
