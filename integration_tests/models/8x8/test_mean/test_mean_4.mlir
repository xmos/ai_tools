// This test reduces the 1st axis of a 3D tensor without keeping dimensions.
func.func @main(%arg0: tensor<10x20x30x!quant.uniform<i8:f32, 0.003:-5>>) -> (tensor<20x30x!quant.uniform<i8:f32, 0.003:-5>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1xi32>, value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<10x20x30x!quant.uniform<i8:f32, 0.003:-5>>, tensor<1xi32>) -> tensor<20x30x!quant.uniform<i8:f32, 0.003:-5>>
  return %1 : tensor<20x30x!quant.uniform<i8:f32, 0.003:-5>>
}
