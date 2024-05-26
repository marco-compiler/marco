// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// Scalar product.

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: tensor<3x4xi64>) -> tensor<3x4xi64>
// CHECK:   %[[splat:.*]] = tensor.splat %[[arg0]] : tensor<3x4xi64>
// CHECK:   %[[destination:.*]] = tensor.empty() : tensor<3x4xi64>
// CHECK:   %[[result:.*]] = linalg.mul ins(%[[arg1]], %[[splat]] : tensor<3x4xi64>, tensor<3x4xi64>) outs(%[[destination]] : tensor<3x4xi64>) -> tensor<3x4xi64>
// CHECK:   return %[[result]]

func.func @foo(%arg0 : i64, %arg1 : tensor<3x4xi64>) -> tensor<3x4xi64> {
    %0 = bmodelica.mul %arg0, %arg1 : (i64, tensor<3x4xi64>) -> tensor<3x4xi64>
    func.return %0 : tensor<3x4xi64>
}

// -----

// Cross product.

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3xi64>, %[[arg1:.*]]: tensor<3xi64>) -> i64
// CHECK:   %[[destination:.*]] = tensor.empty() : tensor<i64>
// CHECK:   %[[tensor_result:.*]] = linalg.dot ins(%[[arg0]], %[[arg1]] : tensor<3xi64>, tensor<3xi64>) outs(%[[destination]] : tensor<i64>) -> tensor<i64>
// CHECK:   %[[result:.*]] = tensor.extract %[[tensor_result]][]
// CHECK:   return %[[result]]

func.func @foo(%arg0 : tensor<3xi64>, %arg1 : tensor<3xi64>) -> i64 {
    %0 = bmodelica.mul %arg0, %arg1 : (tensor<3xi64>, tensor<3xi64>) -> i64
    func.return %0 : i64
}

// -----

// Vector-matrix product.

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3xi64>, %[[arg1:.*]]: tensor<3x2xi64>) -> tensor<2xi64>
// CHECK:   %[[destination:.*]] = tensor.empty() : tensor<2xi64>
// CHECK:   %[[result:.*]] = linalg.vecmat ins(%[[arg0]], %[[arg1]] : tensor<3xi64>, tensor<3x2xi64>) outs(%[[destination]] : tensor<2xi64>) -> tensor<2xi64>
// CHECK:   return %[[result]]

func.func @foo(%arg0 : tensor<3xi64>, %arg1 : tensor<3x2xi64>) -> tensor<2xi64> {
    %0 = bmodelica.mul %arg0, %arg1 : (tensor<3xi64>, tensor<3x2xi64>) -> tensor<2xi64>
    func.return %0 : tensor<2xi64>
}

// -----

// Matrix-vector product.

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x2xi64>, %[[arg1:.*]]: tensor<2xi64>) -> tensor<3xi64>
// CHECK:   %[[transpose_destination:.*]] = tensor.empty() : tensor<2x3xi64>
// CHECK:   %[[transpose:.*]] = linalg.transpose ins(%[[arg0]] : tensor<3x2xi64>) outs(%[[transpose_destination]] : tensor<2x3xi64>) permutation = [1, 0]
// CHECK:   %[[vecmat_destination:.*]] = tensor.empty() : tensor<3xi64>
// CHECK:   %[[result:.*]] = linalg.vecmat ins(%[[arg1]], %[[transpose]] : tensor<2xi64>, tensor<2x3xi64>) outs(%[[vecmat_destination]] : tensor<3xi64>) -> tensor<3xi64>
// CHECK:   return %[[result]]

func.func @foo(%arg0 : tensor<3x2xi64>, %arg1 : tensor<2xi64>) -> tensor<3xi64> {
    %0 = bmodelica.mul %arg0, %arg1 : (tensor<3x2xi64>, tensor<2xi64>) -> tensor<3xi64>
    func.return %0 : tensor<3xi64>
}

// -----

// Matrix product.

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x4xi64>, %[[arg1:.*]]: tensor<4x5xi64>) -> tensor<3x5xi64>
// CHECK:   %[[destination:.*]] = tensor.empty() : tensor<3x5xi64>
// CHECK:   %[[result:.*]] = linalg.matmul ins(%[[arg0]], %[[arg1]] : tensor<3x4xi64>, tensor<4x5xi64>) outs(%[[destination]] : tensor<3x5xi64>) -> tensor<3x5xi64>
// CHECK:   return %[[result]]

func.func @foo(%arg0 : tensor<3x4xi64>, %arg1 : tensor<4x5xi64>) -> tensor<3x5xi64> {
    %0 = bmodelica.mul %arg0, %arg1 : (tensor<3x4xi64>, tensor<4x5xi64>) -> tensor<3x5xi64>
    func.return %0 : tensor<3x5xi64>
}
