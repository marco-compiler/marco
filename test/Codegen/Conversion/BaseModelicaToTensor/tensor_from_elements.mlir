// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64, %[[arg2:.*]]: i64, %[[arg3:.*]]: i64, %[[arg4:.*]]: i64, %[[arg5:.*]]: i64) -> tensor<2x3xi64>
// CHECK: %[[tensor:.*]] = tensor.from_elements %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]] : tensor<2x3xi64>
// CHECK: return %[[tensor]]

func.func @foo(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) -> tensor<2x3xi64> {
    %0 = bmodelica.tensor_from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : i64, i64, i64, i64, i64, i64 -> tensor<2x3xi64>
    func.return %0 : tensor<2x3xi64>
}
