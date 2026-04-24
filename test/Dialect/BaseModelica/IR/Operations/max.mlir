// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Scalars
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real)

func.func @Scalars(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.max %arg0, %arg1 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.max %[[arg0]], %[[arg1]] : !bmodelica.real, !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @Tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?x!bmodelica.real>)

func.func @Tensor(%arg0: tensor<?x?x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.max %arg0 : tensor<?x?x!bmodelica.real> -> !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.max %[[arg0]] : tensor<?x?x!bmodelica.real> -> !bmodelica.real
    // CHECK-NEXT: return %[[result]]
}
