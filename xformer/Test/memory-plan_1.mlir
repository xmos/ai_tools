// RUN: xcore-opt --mlir-io %s --xcore-plan-memory -mlir-print-ir-module-scope -mlir-disable-threading | FileCheck %s

// CHECK: xc.offsets = dense<[384, 0]> : vector<2xi32>
func.func @main(%arg0: tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> {tf_saved_model.index_path = ["zero_padding2d_input"]}) -> (tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>> {tf_saved_model.index_path = ["zero_padding2d"]}) attributes {tf.entry_function = {inputs = "serving_default_zero_padding2d_input:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %0 = "xc.pad"(%arg0) <{end = 0 : i32, num_copies = 3 : i32, pad_size = 96 : i32, size = 48 : i32, start = 96 : i32, use_vpu = true, zero_point = -1 : i32}> : (tensor<1x4x1x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>) -> tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
  return %0 : tensor<1x4x3x48x!quant.uniform<i8:f32, 0.0078384801745414734:-1>>
}