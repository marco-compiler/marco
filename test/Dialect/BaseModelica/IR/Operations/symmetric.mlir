// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x4x!bmodelica.real>)

func.func @StaticDimensions(%arg0: tensor<4x4x!bmodelica.real>) -> tensor<4x4x!bmodelica.real> {
    %0 = bmodelica.symmetric %arg0 : tensor<4x4x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    return %0 : tensor<4x4x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.symmetric %[[arg0]] : tensor<4x4x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicDimensions
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?x!bmodelica.real>)

func.func @DynamicDimensions(%arg0: tensor<?x?x!bmodelica.real>) -> tensor<?x?x!bmodelica.real> {
    %0 = bmodelica.symmetric %arg0 : tensor<?x?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    return %0 : tensor<?x?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.symmetric %[[arg0]] : tensor<?x?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
