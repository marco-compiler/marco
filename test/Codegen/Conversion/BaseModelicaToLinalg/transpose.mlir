// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// CHECK-LABEL: @staticTensors
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4xi64>) -> tensor<4x3xi64>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.transpose ins(%[[arg0]] : tensor<3x4xi64>) outs(%[[destination]] : tensor<4x3xi64>) permutation = [1, 0]
// CHECK:   return %[[result]]

func.func @staticTensors(%arg0 : tensor<3x4xi64>) -> tensor<4x3xi64> {
    %0 = bmodelica.transpose %arg0 : tensor<3x4xi64> -> tensor<4x3xi64>
    func.return %0 : tensor<4x3xi64>
}

// -----

// CHECK-LABEL: @dynamicTensors
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x?xi64>) -> tensor<?x3xi64>
// CHECK:   %[[one:.*]] = arith.constant 1 : index
// CHECK:   %[[dim:.*]] = tensor.dim %[[arg0]], %[[one]]
// CHECK:   %[[destination:.*]] = tensor.empty(%[[dim]])
// CHECK:   %[[result:.*]] = linalg.transpose
// CHECK:   return %[[result]]

func.func @dynamicTensors(%arg0 : tensor<3x?xi64>) -> tensor<?x3xi64> {
    %0 = bmodelica.transpose %arg0 : tensor<3x?xi64> -> tensor<?x3xi64>
    func.return %0 : tensor<?x3xi64>
}
