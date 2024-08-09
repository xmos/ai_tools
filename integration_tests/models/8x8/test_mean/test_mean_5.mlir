// This test reduces all axes of a 2D tensor while keeping dimensions.
func.func @main(%arg0: tensor<5x7x!quant.uniform<i8:f32, 0.002:-3>>) -> (tensor<1x1x!quant.uniform<i8:f32, 0.002:-3>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2xi32>, value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = true} : (tensor<5x7x!quant.uniform<i8:f32, 0.002:-3>>, tensor<2xi32>) -> tensor<1x1x!quant.uniform<i8:f32, 0.002:-3>>
  return %1 : tensor<1x1x!quant.uniform<i8:f32, 0.002:-3>>
}
