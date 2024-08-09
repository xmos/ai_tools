// This test reduces a 1D tensor to a scalar.
func.func @main(%arg0: tensor<15x!quant.uniform<i8:f32, 0.009:0>>) -> (tensor<!quant.uniform<i8:f32, 0.009:0>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1xi32>, value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<15x!quant.uniform<i8:f32, 0.009:0>>, tensor<1xi32>) -> tensor<!quant.uniform<i8:f32, 0.009:0>>
  return %1 : tensor<!quant.uniform<i8:f32, 0.009:0>>
}
