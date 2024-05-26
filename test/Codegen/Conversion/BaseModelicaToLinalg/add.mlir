// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// CHECK: func @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x4x5x6xi64>, %[[arg1:.*]]: tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.add ins(%[[arg0]], %[[arg1]] : tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) outs(%[[destination]] : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
// CHECK:   return %[[result]]

func.func @foo(%arg0 : tensor<3x4x5x6xi64>, %arg1 : tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64> {
    %0 = bmodelica.add %arg0, %arg1 : (tensor<3x4x5x6xi64>, tensor<3x4x5x6xi64>) -> tensor<3x4x5x6xi64>
    func.return %0 : tensor<3x4x5x6xi64>
}
