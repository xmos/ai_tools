// This test reduces the 2nd and 3rd axes of a 4D tensor with consecutive axes and keep_dims = true.
func.func @main(%arg0: tensor<8x5x10x12x!quant.uniform<i8:f32, 0.008:2>>) -> (tensor<8x1x1x12x!quant.uniform<i8:f32, 0.008:2>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2xi32>, value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = true} : (tensor<8x5x10x12x!quant.uniform<i8:f32, 0.008:2>>, tensor<2xi32>) -> tensor<8x1x1x12x!quant.uniform<i8:f32, 0.008:2>>
  return %1 : tensor<8x1x1x12x!quant.uniform<i8:f32, 0.008:2>>
}
