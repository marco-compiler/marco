// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mprint_void_ai64(tensor<*xi64>)

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi64>)
// CHECK: %[[cast:.*]] = tensor.cast %[[arg0]] : tensor<2x3xi64> to tensor<*xi64>
// CHECK: runtime.call @_Mprint_void_ai64(%[[cast]])
// CHECK: return

func.func @foo(%arg0: tensor<2x3xi64>) {
    bmodelica.print %arg0 : tensor<2x3xi64>
    func.return
}
