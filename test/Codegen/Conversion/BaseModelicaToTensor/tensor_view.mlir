// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// Scalar subscripts.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x3x2xi64>, %[[arg1:.*]]: index) -> tensor<3x2xi64>
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[arg0]][%[[arg1]], 0, 0] [1, 3, 2] [1, 1, 1] : tensor<4x3x2xi64> to tensor<3x2xi64>
// CHECK: return %[[slice]]

func.func @foo(%arg0: tensor<4x3x2xi64>, %arg1: index) -> tensor<3x2xi64> {
    %0 = bmodelica.tensor_view %arg0[%arg1] : tensor<4x3x2xi64>, index -> tensor<3x2xi64>
    func.return %0 : tensor<3x2xi64>
}

// -----

// Range subscripts.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x3x2xi64>, %[[arg1:.*]]: !bmodelica<range index>) -> tensor<?x3x2xi64>
// CHECK-DAG: %[[begin:.*]] = bmodelica.range_begin %[[arg1]]
// CHECK-DAG: %[[size:.*]] = bmodelica.range_size %[[arg1]]
// CHECK-DAG: %[[step:.*]] = bmodelica.range_step %[[arg1]]
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[arg0]][%[[begin]], 0, 0] [%[[size]], 3, 2] [%[[step]], 1, 1] : tensor<4x3x2xi64> to tensor<?x3x2xi64>
// CHECK: return %[[slice]]

func.func @foo(%arg0: tensor<4x3x2xi64>, %arg1: !bmodelica<range index>) -> tensor<?x3x2xi64> {
    %0 = bmodelica.tensor_view %arg0[%arg1] : tensor<4x3x2xi64>, !bmodelica<range index> -> tensor<?x3x2xi64>
    func.return %0 : tensor<?x3x2xi64>
}
