// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mtranspose_void_ai64_ai64(memref<*xi64>, tensor<*xi64>)

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi64>) -> tensor<3x2xi64>
// CHECK: %[[alloc:.*]] = memref.alloc() : memref<3x2xi64>
// CHECK: %[[cast_1:.*]] = memref.cast %[[alloc]] : memref<3x2xi64> to memref<*xi64>
// CHECK: %[[cast_2:.*]] = tensor.cast %[[arg0]] : tensor<2x3xi64> to tensor<*xi64>
// CHECK: runtime.call @_Mtranspose_void_ai64_ai64(%[[cast_1]], %[[cast_2]])
// CHECK: %[[cast_3:.*]] = memref.cast %[[cast_1]] : memref<*xi64> to memref<3x2xi64>
// CHECK: %[[to_tensor:.*]] = bufferization.to_tensor %[[cast_3]]
// CHECK: return %[[to_tensor]]

func.func @foo(%arg0: tensor<2x3xi64>) -> tensor<3x2xi64> {
    %0 = bmodelica.transpose %arg0 : tensor<2x3xi64> -> tensor<3x2xi64>
    func.return %0 : tensor<3x2xi64>
}
