// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>)

func.func @StaticDimensions(%arg0: tensor<2x3x!bmodelica.real>) -> index {
    %0 = bmodelica.ndims %arg0 : tensor<2x3x!bmodelica.real> -> index
    return %0 : index

    // CHECK: %[[result:.*]] = bmodelica.ndims %[[arg0]] : tensor<2x3x!bmodelica.real> -> index
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?x!bmodelica.real>)

func.func @DynamicDimensions(%arg0: tensor<?x?x!bmodelica.real>) -> index {
    %0 = bmodelica.ndims %arg0 : tensor<?x?x!bmodelica.real> -> index
    return %0 : index

    // CHECK: %[[result:.*]] = bmodelica.ndims %[[arg0]] : tensor<?x?x!bmodelica.real> -> index
    // CHECK-NEXT: return %[[result]]
}
