// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_MminScalars_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @scalars
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = runtime.call @_MminScalars_f64_f64_f64(%[[arg0]], %[[arg1]])
// CHECK: return %[[result]]

func.func @scalars(%arg0: f64, %arg1: f64) -> f64 {
    %0 = bmodelica.min %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}

// -----

// CHECK: runtime.function private @_MminArray_i64_ai64(tensor<*xi64>) -> i64

// CHECK-LABEL: @tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi64>) -> i64
// CHECK: %[[cast:.*]] = tensor.cast %[[arg0]] : tensor<2x3xi64> to tensor<*xi64>
// CHECK: %[[result:.*]] = runtime.call @_MminArray_i64_ai64(%[[cast]])
// CHECK: return %[[result]]

func.func @tensor(%arg0: tensor<2x3xi64>) -> i64 {
    %0 = bmodelica.min %arg0 : tensor<2x3xi64> -> i64
    func.return %0 : i64
}
