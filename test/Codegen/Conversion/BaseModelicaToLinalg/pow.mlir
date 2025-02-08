// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg --cse | FileCheck %s

// CHECK-LABEL: @matrixBase
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x3xf64>, %[[arg1:.*]]: index) -> tensor<3x3xf64>
// CHECK-DAG:   %[[one_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG:   %[[one_f64:.*]] = bmodelica.cast %[[one_i64]] : i64 -> f64
// CHECK-DAG:   %[[init:.*]] = tensor.splat %[[one_f64]] : tensor<3x3xf64>
// CHECK-DAG:   %[[lowerBound:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       %[[result:.*]] = scf.for %[[i:.*]] to %[[arg1]] step %[[step]] iter_args(%[[it:.*]] = %[[init]]) -> (tensor<3x3xf64>) {
// CHECK:           %[[zero_i64:.*]] = arith.constant 0 : i64
// CHECK:           %[[zero_f64:.*]] = bmodelica.cast %[[zero_i64]] : i64 -> f64
// CHECK:           %[[destination:.*]] = tensor.splat %[[zero_f64]] : tensor<3x3xf64>
// CHECK:           %[[matmul:.*]] = linalg.matmul ins(%[[it]], %[[arg0]] : tensor<3x3xf64>, tensor<3x3xf64>) outs(%[[destination]] : tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK:           scf.yield %[[matmul]]
// CHECK:       }
// CHECK:       return %[[result]]

func.func @matrixBase(%arg0 : tensor<3x3xf64>, %arg1 : index) -> tensor<3x3xf64> {
    %0 = bmodelica.pow %arg0, %arg1 : (tensor<3x3xf64>, index) -> tensor<3x3xf64>
    func.return %0 : tensor<3x3xf64>
}
