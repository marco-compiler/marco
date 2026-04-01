// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Scalar
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)

func.func @Scalar(%arg0: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.ceil %arg0 : !bmodelica.real -> !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.ceil %[[arg0]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @Tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>)

func.func @Tensor(%arg0: tensor<2x3x!bmodelica.real>) -> tensor<2x3x!bmodelica.real> {
    %0 = bmodelica.ceil %arg0 : tensor<2x3x!bmodelica.real> -> tensor<2x3x!bmodelica.real>
    return %0 : tensor<2x3x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.ceil %[[arg0]] : tensor<2x3x!bmodelica.real> -> tensor<2x3x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
