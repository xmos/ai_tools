// RUN: xcore-opt --mlir-io %s --xcore-replace-broadcast | FileCheck %s 

// CHECK-LABEL: valid_broadcast1
func.func @valid_broadcast1(%arg0: tensor<1x5x1x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.broadcast
  // CHECK-NOT: tfl.broadcast_to
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 16]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x5x1x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

// CHECK-LABEL: valid_broadcast2
func.func @valid_broadcast2(%arg0: tensor<1x5x8x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.broadcast
  // CHECK-NOT: tfl.broadcast_to
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 16]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x5x8x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x16x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

// CHECK-LABEL: valid_broadcast3
func.func @valid_broadcast3(%arg0: tensor<1x5x8x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.broadcast
  // CHECK-NOT: tfl.broadcast_to
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 23]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x5x8x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

// CHECK-LABEL: valid_broadcast4
func.func @valid_broadcast4(%arg0: tensor<1x5x1x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.broadcast
  // CHECK-NOT: tfl.broadcast_to
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 23]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x5x1x1x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

// CHECK-LABEL: valid_broadcast5
func.func @valid_broadcast5(%arg0: tensor<8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>> {tf_saved_model.index_path = ["tf.broadcast_to_1"]}) attributes {tf.entry_function = {inputs = "serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: xc.broadcast
  // CHECK-NOT: tfl.broadcast_to
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4xi32>, value = dense<[1, 5, 8, 23]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>, tensor<4xi32>) -> tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
  return %1 : tensor<1x5x8x23x!quant.uniform<i8:f32, 0.0078426999971270561:-1>>
}

