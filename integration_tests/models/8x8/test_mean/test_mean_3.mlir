// This test reduces the 2nd and 4th axes of a 5D tensor while keeping dimensions.
func.func @main(%arg0: tensor<4x3x5x7x6x!quant.uniform<i8:f32, 0.0045:0>>) -> (tensor<4x1x5x1x6x!quant.uniform<i8:f32, 0.0045:0>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2xi32>, value = dense<[1, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = true} : (tensor<4x3x5x7x6x!quant.uniform<i8:f32, 0.0045:0>>, tensor<2xi32>) -> tensor<4x1x5x1x6x!quant.uniform<i8:f32, 0.0045:0>>
  return %1 : tensor<4x1x5x1x6x!quant.uniform<i8:f32, 0.0045:0>>
}
