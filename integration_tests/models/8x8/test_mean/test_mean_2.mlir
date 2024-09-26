// This test reduces the 3rd axis of a 4D tensor without keeping dimensions.
func.func @main(%arg0: tensor<2x3x4x5x!quant.uniform<i8:f32, 0.005:-128>>) -> (tensor<2x3x5x!quant.uniform<i8:f32, 0.006:-127>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1xi32>, value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<2x3x4x5x!quant.uniform<i8:f32, 0.005:-128>>, tensor<1xi32>) -> tensor<2x3x5x!quant.uniform<i8:f32, 0.006:-127>>
  return %1 : tensor<2x3x5x!quant.uniform<i8:f32, 0.006:-127>>
}
