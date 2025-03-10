func.func @main(%arg0: tensor<3x4x5x6x7x!quant.uniform<i16:f32, 0.15>>) -> (tensor<4x5x6x7x3x!quant.uniform<i16:f32, 0.15>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<5xi32>, value = dense<[0, 4, 3, 2, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tfl.transpose"(%arg0, %0) : (tensor<3x4x5x6x7x!quant.uniform<i16:f32, 0.15>>, tensor<5xi32>) -> tensor<3x7x6x5x4x!quant.uniform<i16:f32, 0.15>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<5xi32>, value = dense<[4, 3, 2, 1, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
  %3 = "tfl.transpose"(%1, %2) : (tensor<3x7x6x5x4x!quant.uniform<i16:f32, 0.15>>, tensor<5xi32>) -> tensor<4x5x6x7x3x!quant.uniform<i16:f32, 0.15>>
  return %3 : tensor<4x5x6x7x3x!quant.uniform<i16:f32, 0.15>>
}
