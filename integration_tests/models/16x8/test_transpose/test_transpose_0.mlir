func.func @main(%arg0: tensor<4x6x5x8x!quant.uniform<i16:f32, 0.01>>) -> (tensor<4x5x6x8x!quant.uniform<i16:f32, 0.01>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.transpose"(%arg0, %0) : (tensor<4x6x5x8x!quant.uniform<i16:f32, 0.01>>, tensor<4xi32>) -> tensor<4x5x6x8x!quant.uniform<i16:f32, 0.01>>
  return %1 : tensor<4x5x6x8x!quant.uniform<i16:f32, 0.01>>
}
