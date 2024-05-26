// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mlinspace_void_af64_f64_f64(memref<*xf64>, f64, f64)

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64, %[[arg2:.*]]: index) -> tensor<?xf64>
// CHECK: %[[alloc:.*]] = memref.alloc(%[[arg2]]) : memref<?xf64>
// CHECK: %[[cast_1:.*]] = memref.cast %[[alloc]] : memref<?xf64> to memref<*xf64>
// CHECK: runtime.call @_Mlinspace_void_af64_f64_f64(%[[cast_1]], %[[arg0]], %[[arg1]])
// CHECK: %[[cast_2:.*]] = memref.cast %[[cast_1]] : memref<*xf64> to memref<?xf64>
// CHECK: %[[to_tensor:.*]] = bufferization.to_tensor %[[cast_2]]
// CHECK: return %[[to_tensor]]

func.func @foo(%arg0: f64, %arg1: f64, %arg2: index) -> tensor<?xf64> {
    %0 = bmodelica.linspace %arg0, %arg1, %arg2 : (f64, f64, index) -> tensor<?xf64>
    func.return %0 : tensor<?xf64>
}
