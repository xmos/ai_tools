func.func @main(%arg0: tensor<2x3x4x5x6x!quant.uniform<i16:f32, 0.1>>) -> (tensor<2x6x5x4x3x!quant.uniform<i16:f32, 0.1>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<5xi32>, value = dense<[0, 4, 3, 2, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tfl.transpose"(%arg0, %0) : (tensor<2x3x4x5x6x!quant.uniform<i16:f32, 0.1>>, tensor<5xi32>) -> tensor<2x6x5x4x3x!quant.uniform<i16:f32, 0.1>>
  return %1 : tensor<2x6x5x4x3x!quant.uniform<i16:f32, 0.1>>
}

