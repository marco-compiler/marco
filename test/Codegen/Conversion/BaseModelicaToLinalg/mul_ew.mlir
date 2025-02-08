// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// CHECK-LABEL: @tensorTensor
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4x5x6xi64>, %[[arg1:.*]]: tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.mul ins(%[[arg0]], %[[arg1]] : tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) outs(%[[destination]] : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   return %[[result]]

func.func @tensorTensor(%arg0 : tensor<3x4x5x6xi64>, %arg1 : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64> {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
    func.return %0 : tensor<3x4x5x6xi64>
}

// -----

// CHECK-LABEL: @scalarTensor
// CHECK-SAME:  (%[[arg0:.*]]: i64, %[[arg1:.*]]: tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   %[[splat:.*]] = tensor.splat %[[arg0]] : tensor<3x4x5x6xi64>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.mul ins(%[[arg1]], %[[splat]] : tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) outs(%[[destination]] : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   return %[[result]]

func.func @scalarTensor(%arg0 : i64, %arg1 : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64> {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (i64, tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
    func.return %0 : tensor<3x4x5x6xi64>
}

// -----

// CHECK-LABEL: @tensorScalar
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4x5x6xi64>, %[[arg1:.*]]: i64) -> tensor<3x4x5x6xi64>
// CHECK:   %[[splat:.*]] = tensor.splat %[[arg1]] : tensor<3x4x5x6xi64>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.mul ins(%[[arg0]], %[[splat]] : tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) outs(%[[destination]] : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   return %[[result]]

func.func @tensorScalar(%arg0 : tensor<3x4x5x6xi64>, %arg1 : i64) -> tensor<3x4x5x6xi64> {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (tensor<3x4x5x6xi64>, i64) -> tensor<3x4x5x6xi64>
    func.return %0 : tensor<3x4x5x6xi64>
}
