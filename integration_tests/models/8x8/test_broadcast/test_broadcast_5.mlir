func.func @main(%arg0: tensor<!quant.uniform<i8:f32, 6.5359479049220681E-4:-128>>) -> tensor<1x1x13x64x!quant.uniform<i8:f32, 6.5359479049220681E-4:-128>> attributes {tf.entry_function = {inputs = "arg0", outputs = "0"}} {
  %cst = arith.constant dense<[1, 1, 13, 64]> : tensor<4xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<!quant.uniform<i8:f32, 6.5359479049220681E-4:-128>>, tensor<4xi32>) -> tensor<1x1x13x64x!quant.uniform<i8:f32, 6.5359479049220681E-4:-128>>
  return %0 : tensor<1x1x13x64x!quant.uniform<i8:f32, 6.5359479049220681E-4:-128>>
}
