// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>)

func.func @StaticDimensions(%arg0: tensor<2x3x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.product %arg0 : tensor<2x3x!bmodelica.real> -> !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.product %[[arg0]] : tensor<2x3x!bmodelica.real> -> !bmodelica.real
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?x!bmodelica.real>)

func.func @DynamicDimensions(%arg0: tensor<?x?x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.product %arg0 : tensor<?x?x!bmodelica.real> -> !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.product %[[arg0]] : tensor<?x?x!bmodelica.real> -> !bmodelica.real
    // CHECK-NEXT: return %[[result]]
}
