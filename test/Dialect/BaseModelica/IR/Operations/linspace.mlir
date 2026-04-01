// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real, %[[arg2:.*]]: !bmodelica.real)

func.func @StaticDimensions(%arg0: !bmodelica.real, %arg1: !bmodelica.real, %arg2: !bmodelica.real) -> tensor<10x!bmodelica.real> {
    %0 = bmodelica.linspace %arg0, %arg1, %arg2 : !bmodelica.real, !bmodelica.real, !bmodelica.real -> tensor<10x!bmodelica.real>
    return %0 : tensor<10x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.linspace %[[arg0]], %[[arg1]], %[[arg2]] : !bmodelica.real, !bmodelica.real, !bmodelica.real -> tensor<10x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real, %[[arg2:.*]]: !bmodelica.real)

func.func @DynamicDimensions(%arg0: !bmodelica.real, %arg1: !bmodelica.real, %arg2: !bmodelica.real) -> tensor<?x!bmodelica.real> {
    %0 = bmodelica.linspace %arg0, %arg1, %arg2 : !bmodelica.real, !bmodelica.real, !bmodelica.real -> tensor<?x!bmodelica.real>
    return %0 : tensor<?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.linspace %[[arg0]], %[[arg1]], %[[arg2]] : !bmodelica.real, !bmodelica.real, !bmodelica.real -> tensor<?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
