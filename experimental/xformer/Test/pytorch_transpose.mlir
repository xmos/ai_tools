// RUN: xcore-opt --mlir-io %s --xcore-optimize-transpose | FileCheck %s
// RUN: xcore-opt --mlir-io %s --xcore-optimize-transpose --xcore-allow-input-modification | FileCheck %s -check-prefix=INPUT-CHECK

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

// -----

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

// -----

// INPUT-CHECK-LABEL: fold_to_input
// INPUT-CHECK: %arg0: tensor<?x45x80x16
func.func @fold_to_input(%arg0: tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %10 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %12 = "tfl.pseudo_const"() {value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // INPUT-CHECK-NOT: transpose
  %19 = "tfl.transpose"(%arg0, %11) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %20 = "tfl.pad"(%19, %12) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4x2xi32>) -> tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %20 : tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}

// -----

// CHECK-LABEL: fold_to_input2
// CHECK: %arg0: tensor<?x16x45x80
func.func @fold_to_input2(%arg0: tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) -> (tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>) {
  %10 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %11 = "tfl.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %12 = "tfl.pseudo_const"() {value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // CHECK: transpose
  %19 = "tfl.transpose"(%arg0, %11) : (tensor<?x16x45x80x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4xi32>) -> tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  %20 = "tfl.pad"(%19, %12) : (tensor<?x45x80x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>, tensor<4x2xi32>) -> tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
  return %20 : tensor<?x47x82x16x!quant.uniform<i8:f32, 0.13334976136684418:-128>>
}
