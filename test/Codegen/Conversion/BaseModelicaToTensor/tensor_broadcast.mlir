// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64) -> tensor<2x3xi64>
// CHECK: %[[tensor:.*]] = tensor.splat %[[arg0]] : tensor<2x3xi64>
// CHECK: return %[[tensor]]

func.func @foo(%arg0: i64) -> tensor<2x3xi64> {
    %0 = bmodelica.tensor_broadcast %arg0 : i64 -> tensor<2x3xi64>
    func.return %0 : tensor<2x3xi64>
}
