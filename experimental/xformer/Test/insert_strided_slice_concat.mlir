// RUN: xcore-opt --mlir-io --xcore-opsplit %s

func.func @main(%arg0: tensor<?x6x6x4x!quant.uniform<i8:f32, 0.0039214449934661388:-128>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<?x144x!quant.uniform<i8:f32, 0.0032160764094442129:-128>> {tf_saved_model.index_path = ["flatten_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: tfl.strided_slice
  // CHECK: tfl.strided_slice
  // CHECK: tfl.concatenation
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4x1x1x4x!quant.uniform<i8<-127:127>:f32:0, {0.0057157878763973713,0.0063750739209353924,0.0039762118831276894,0.0061128144152462482}>>, value = dense<[[[[-17, 25, -70, -127]]], [[[24, 51, 11, -127]]], [[[-127, 22, -2, -44]]], [[[127, -84, 14, -67]]]]> : tensor<4x1x1x4xi8>} : () -> tensor<4x1x1x4x!quant.uniform<i8<-127:127>:f32:0, {0.0057157878763973713,0.0063750739209353924,0.0039762118831276894,0.0061128144152462482}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<4x!quant.uniform<i32:f32:0, {2.2414147679228336E-5,2.4999500965350308E-5,1.5592495401506312E-5,2.3971066184458323E-5}>>, value = dense<0> : tensor<4xi32>} : () -> tensor<4x!quant.uniform<i32:f32:0, {2.2414147679228336E-5,2.4999500965350308E-5,1.5592495401506312E-5,2.3971066184458323E-5}>>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x6x6x4x!quant.uniform<i8:f32, 0.0039214449934661388:-128>>, tensor<4x1x1x4x!quant.uniform<i8<-127:127>:f32:0, {0.0057157878763973713,0.0063750739209353924,0.0039762118831276894,0.0061128144152462482}>>, tensor<4x!quant.uniform<i32:f32:0, {2.2414147679228336E-5,2.4999500965350308E-5,1.5592495401506312E-5,2.3971066184458323E-5}>>) -> tensor<?x6x6x4x!quant.uniform<i8:f32, 0.0032160764094442129:-128>>
  %3 = "tfl.pseudo_const"() {value = dense<[-1, 144]> : tensor<2xi32>} : () -> tensor<2xi32>
  %4 = "tfl.reshape"(%2, %3) : (tensor<?x6x6x4x!quant.uniform<i8:f32, 0.0032160764094442129:-128>>, tensor<2xi32>) -> tensor<?x144x!quant.uniform<i8:f32, 0.0032160764094442129:-128>>
  return %4 : tensor<?x144x!quant.uniform<i8:f32, 0.0032160764094442129:-128>>
}
