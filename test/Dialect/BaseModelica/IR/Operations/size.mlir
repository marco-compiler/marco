// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @AllDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>)

func.func @AllDimensions(%arg0: tensor<2x3x!bmodelica.real>) -> tensor<2x!bmodelica.real> {
    %0 = bmodelica.size %arg0 : tensor<2x3x!bmodelica.real> -> tensor<2x!bmodelica.real>
    return %0 : tensor<2x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.size %[[arg0]] : tensor<2x3x!bmodelica.real> -> tensor<2x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @SingleDimension
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @SingleDimension(%arg0: tensor<2x3x!bmodelica.real>, %arg1: index) -> index {
    %0 = bmodelica.size %arg0, %arg1 : tensor<2x3x!bmodelica.real>, index -> index
    return %0 : index

    // CHECK: %[[result:.*]] = bmodelica.size %[[arg0]], %[[arg1]] : tensor<2x3x!bmodelica.real>, index -> index
    // CHECK-NEXT: return %[[result]]
}
