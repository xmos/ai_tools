// This test reduces the 2nd and 3rd axes of a 3D tensor with different input/output quantization parameters.
func.func @main(%arg0: tensor<5x6x7x!quant.uniform<i8:f32, 0.004:-2>>) -> (tensor<5x!quant.uniform<i8:f32, 0.0035:-1>>) {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2xi32>, value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<5x6x7x!quant.uniform<i8:f32, 0.004:-2>>, tensor<2xi32>) -> tensor<5x!quant.uniform<i8:f32, 0.0035:-1>>
  return %1 : tensor<5x!quant.uniform<i8:f32, 0.0035:-1>>
}
