func.func @main(%arg0: tensor<2x8x4x6x!quant.uniform<i16:f32, 0.05>>) -> (tensor<4x2x6x8x!quant.uniform<i16:f32, 0.05>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 0, 3, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.transpose"(%arg0, %0) : (tensor<2x8x4x6x!quant.uniform<i16:f32, 0.05>>, tensor<4xi32>) -> tensor<8x2x6x4x!quant.uniform<i16:f32, 0.05>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tfl.transpose"(%1, %2) : (tensor<8x2x6x4x!quant.uniform<i16:f32, 0.05>>, tensor<4xi32>) -> tensor<8x6x4x2x!quant.uniform<i16:f32, 0.05>>
  %4 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[2, 3, 1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
  %5 = "tfl.transpose"(%3, %4) : (tensor<8x6x4x2x!quant.uniform<i16:f32, 0.05>>, tensor<4xi32>) -> tensor<4x2x6x8x!quant.uniform<i16:f32, 0.05>>
  return %5 : tensor<4x2x6x8x!quant.uniform<i16:f32, 0.05>>
}

