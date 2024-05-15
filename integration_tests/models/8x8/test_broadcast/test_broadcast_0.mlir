func.func @main(%arg0: tensor<1x5x1x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 16]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x5x1x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

