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

// CHECK-LABEL: move_transpose_over_slice
func.func @move_transpose_over_slice(%arg0: tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x40x7x4x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  // CHECK-NOT: transpose
  // CHECK: slice
  // CHECK-NOT: transpose
  %10 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %12 = "tfl.pseudo_const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %13 = "tfl.pseudo_const"() {value = dense<[-1, 7, 4, 40]> : tensor<4xi32>} : () -> tensor<4xi32>
  %18 = "tfl.transpose"(%arg0, %10) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x80x16x45x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %19 = "tfl.slice"(%18, %12, %13) : (tensor<?x80x16x45x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x7x4x40x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %20 = "tfl.transpose"(%19, %11) : (tensor<?x7x4x40x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x40x7x4x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %20 : tensor<?x40x7x4x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}