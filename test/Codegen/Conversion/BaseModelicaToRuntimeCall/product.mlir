// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mproduct_i64_ai64(tensor<*xi64>) -> i64

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi64>) -> i64
// CHECK: %[[cast:.*]] = tensor.cast %[[arg0]] : tensor<2x3xi64> to tensor<*xi64>
// CHECK: %[[result:.*]] = runtime.call @_Mproduct_i64_ai64(%[[cast]])
// CHECK: return %[[result]]

func.func @foo(%arg0: tensor<2x3xi64>) -> i64 {
    %0 = bmodelica.product %arg0 : tensor<2x3xi64> -> i64
    func.return %0 : i64
}
