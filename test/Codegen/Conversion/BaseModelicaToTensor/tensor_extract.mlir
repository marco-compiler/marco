// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi64>, %[[arg1:.*]]: index, %[[arg2:.*]]: index) -> i64
// CHECK: %[[value:.*]] = tensor.extract %[[arg0]][%[[arg1]], %[[arg2]]]
// CHECK: return %[[value]]

func.func @foo(%arg0: tensor<2x3xi64>, %arg1: index, %arg2: index) -> i64 {
    %0 = bmodelica.tensor_extract %arg0[%arg1, %arg2] : tensor<2x3xi64>, index, index
    func.return %0 : i64
}
