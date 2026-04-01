// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3x!bmodelica.real>)

func.func @StaticDimensions(%arg0: tensor<2x3x!bmodelica.real>) -> tensor<3x2x!bmodelica.real> {
    %0 = bmodelica.transpose %arg0 : tensor<2x3x!bmodelica.real> -> tensor<3x2x!bmodelica.real>
    return %0 : tensor<3x2x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.transpose %[[arg0]] : tensor<2x3x!bmodelica.real> -> tensor<3x2x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?x!bmodelica.real>)

func.func @DynamicDimensions(%arg0: tensor<?x?x!bmodelica.real>) -> tensor<?x?x!bmodelica.real> {
    %0 = bmodelica.transpose %arg0 : tensor<?x?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    return %0 : tensor<?x?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.transpose %[[arg0]] : tensor<?x?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
