// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Msymmetric_void_ai64_ai64(memref<*xi64>, tensor<*xi64>)

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK: %[[alloc:.*]] = memref.alloc() : memref<3x3xi64>
// CHECK: %[[cast_1:.*]] = memref.cast %[[alloc]] : memref<3x3xi64> to memref<*xi64>
// CHECK: %[[cast_2:.*]] = tensor.cast %[[arg0]] : tensor<3x3xi64> to tensor<*xi64>
// CHECK: runtime.call @_Msymmetric_void_ai64_ai64(%[[cast_1]], %[[cast_2]])
// CHECK: %[[cast_3:.*]] = memref.cast %[[cast_1]] : memref<*xi64> to memref<3x3xi64>
// CHECK: %[[to_tensor:.*]] = bufferization.to_tensor %[[cast_3]]
// CHECK: return %[[to_tensor]]

func.func @foo(%arg0: tensor<3x3xi64>) -> tensor<3x3xi64> {
    %0 = bmodelica.symmetric %arg0 : tensor<3x3xi64> -> tensor<3x3xi64>
    func.return %0 : tensor<3x3xi64>
}
