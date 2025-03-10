func.func @main(%arg0: tensor<3x7x6x4x!quant.uniform<i16:f32, 0.02>>) -> (tensor<4x3x7x6x!quant.uniform<i16:f32, 0.02>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.transpose"(%arg0, %0) : (tensor<3x7x6x4x!quant.uniform<i16:f32, 0.02>>, tensor<4xi32>) -> tensor<3x4x6x7x!quant.uniform<i16:f32, 0.02>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 0, 3, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tfl.transpose"(%1, %2) : (tensor<3x4x6x7x!quant.uniform<i16:f32, 0.02>>, tensor<4xi32>) -> tensor<4x3x7x6x!quant.uniform<i16:f32, 0.02>>
  return %3 : tensor<4x3x7x6x!quant.uniform<i16:f32, 0.02>>
}

