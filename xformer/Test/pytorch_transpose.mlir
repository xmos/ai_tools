// RUN: xcore-opt --mlir-io %s --xcore-optimize-transpose | FileCheck %s

// CHECK-LABEL: hoist_pad_above_transpose
func.func @hoist_pad_above_transpose(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %10 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %12 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: pad
  // CHECK-NOT: transpose
  %18 = "tfl.transpose"(%arg0, %10) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %20 = "tfl.pad"(%18, %12) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4x2xi32>) -> tensor<?x16x47x82x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %19 = "tfl.transpose"(%20, %11) : (tensor<?x16x47x82x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %19 : tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// CHECK-LABEL: fold_cancellable_transpose
func.func @fold_cancellable_transpose(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %10 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %12 = "tfl.pseudo_const"() {value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK-NOT: transpose
  // CHECK: pad
  %18 = "tfl.transpose"(%arg0, %10) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %19 = "tfl.transpose"(%18, %11) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %20 = "tfl.pad"(%19, %12) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4x2xi32>) -> tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %20 : tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// CHECK-LABEL: merge_consecutive_transposes
func.func @merge_consecutive_transposes(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  // CHECK: dense<[0, 1, 3, 2]>
  // CHECK: transpose
  // CHECK-NOT: transpose
  %10 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %18 = "tfl.transpose"(%arg0, %10) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x80x45x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %19 = "tfl.transpose"(%18, %11) : (tensor<?x16x80x45x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %19 : tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// CHECK-LABEL: func.func @erase_double_transpose_three_children(
// CHECK-SAME: [[THREE_INPUT:%[^:]+]]:
// CHECK-NOT: tfl.transpose
// CHECK: return [[THREE_INPUT]], [[THREE_INPUT]], [[THREE_INPUT]] :
// CHECK-NOT: tfl.transpose
func.func @erase_double_transpose_three_children(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %parent_perm = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %inverse_perm = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %parent = "tfl.transpose"(%arg0, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %child0 = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %child1 = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %child2 = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %child0, %child1, %child2 :
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// CHECK-LABEL: func.func @erase_double_transpose_mixed_children(
// CHECK-SAME: [[MIXED_INPUT:%[^:]+]]:
// CHECK-NOT: tfl.transpose
// CHECK: [[MIXED_PERM:%[^ ]+]] = {{.*}}dense<[0, 2, 1, 3]>
// CHECK-NOT: tfl.transpose
// CHECK: [[MIXED_RESULT:%[^ ]+]] = "tfl.transpose"([[MIXED_INPUT]], [[MIXED_PERM]])
// CHECK-NOT: tfl.transpose
// CHECK: return [[MIXED_INPUT]], [[MIXED_RESULT]] :
// CHECK-NOT: tfl.transpose
func.func @erase_double_transpose_mixed_children(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %parent_perm = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %inverse_perm = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %other_perm = "tfl.pseudo_const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %parent = "tfl.transpose"(%arg0, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %inverse_child = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %other_child = "tfl.transpose"(%parent, %other_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %inverse_child, %other_child :
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x80x45x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// CHECK-LABEL: func.func @erase_double_transpose_nested_forks(
// CHECK-SAME: [[NESTED_INPUT:%[^:]+]]:
// CHECK-NOT: tfl.transpose
// CHECK: return [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]], [[NESTED_INPUT]] :
// CHECK-NOT: tfl.transpose
func.func @erase_double_transpose_nested_forks(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
    tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %parent_perm = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %inverse_perm = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %parent = "tfl.transpose"(%arg0, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level1_left = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level1_right = "tfl.transpose"(%parent, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level2_left0 = "tfl.transpose"(%level1_left, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level2_left1 = "tfl.transpose"(%level1_left, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level2_right0 = "tfl.transpose"(%level1_right, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %level2_right1 = "tfl.transpose"(%level1_right, %parent_perm) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf0 = "tfl.transpose"(%level2_left0, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf1 = "tfl.transpose"(%level2_left0, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf2 = "tfl.transpose"(%level2_left1, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf3 = "tfl.transpose"(%level2_left1, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf4 = "tfl.transpose"(%level2_right0, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf5 = "tfl.transpose"(%level2_right0, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf6 = "tfl.transpose"(%level2_right1, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %leaf7 = "tfl.transpose"(%level2_right1, %inverse_perm) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %leaf0, %leaf1, %leaf2, %leaf3, %leaf4, %leaf5, %leaf6, %leaf7 :
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>,
      tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}
