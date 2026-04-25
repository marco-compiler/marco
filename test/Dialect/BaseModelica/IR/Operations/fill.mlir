// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticSizes
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)

func.func @StaticSizes(%arg0: !bmodelica.real) -> tensor<4x!bmodelica.real> {
    %0 = bmodelica.fill %arg0 : !bmodelica.real -> tensor<4x!bmodelica.real>
    return %0 : tensor<4x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.fill %[[arg0]] : !bmodelica.real -> tensor<4x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
