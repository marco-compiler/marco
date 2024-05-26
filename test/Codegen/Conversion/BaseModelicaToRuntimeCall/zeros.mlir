// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mzeros_void_ai64(memref<*xi64>)

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> tensor<?x?xi64>
// CHECK: %[[alloc:.*]] = memref.alloc(%[[arg0]], %[[arg1]]
// CHECK: %[[cast_1:.*]] = memref.cast %[[alloc]] : memref<?x?xi64> to memref<*xi64>
// CHECK: runtime.call @_Mzeros_void_ai64(%[[cast_1]])
// CHECK: %[[cast_2:.*]] = memref.cast %[[cast_1]] : memref<*xi64> to memref<?x?xi64>
// CHECK: %[[to_tensor:.*]] = bufferization.to_tensor %[[cast_2]]
// CHECK: return %[[to_tensor]]

func.func @foo(%arg0: index, %arg1: index) -> tensor<?x?xi64> {
    %0 = bmodelica.zeros %arg0, %arg1 : (index, index) -> tensor<?x?xi64>
    func.return %0 : tensor<?x?xi64>
}
