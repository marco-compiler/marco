// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index, %[[arg2:.*]]: index)

func.func @StaticDimensions(%arg0: index, %arg1: index, %arg2: index) -> tensor<4x4x4x!bmodelica.real> {
    %0 = bmodelica.zeros %arg0, %arg1, %arg2 : index, index, index -> tensor<4x4x4x!bmodelica.real>
    return %0 : tensor<4x4x4x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.zeros %[[arg0]], %[[arg1]], %[[arg2]] : index, index, index -> tensor<4x4x4x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index, %[[arg2:.*]]: index)

func.func @DynamicDimensions(%arg0: index, %arg1: index, %arg2: index) -> tensor<?x?x?x!bmodelica.real> {
    %0 = bmodelica.zeros %arg0, %arg1, %arg2 : index, index, index -> tensor<?x?x?x!bmodelica.real>
    return %0 : tensor<?x?x?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.zeros %[[arg0]], %[[arg1]], %[[arg2]] : index, index, index -> tensor<?x?x?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
