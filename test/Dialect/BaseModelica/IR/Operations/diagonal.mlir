// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @StaticInputStaticResult
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x!bmodelica.real>)

func.func @StaticInputStaticResult(%arg0: tensor<4x!bmodelica.real>) -> tensor<4x4x!bmodelica.real> {
    %0 = bmodelica.diagonal %arg0 : tensor<4x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    return %0 : tensor<4x4x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.diagonal %[[arg0]] : tensor<4x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @StaticInputDynamicResult
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x!bmodelica.real>)

func.func @StaticInputDynamicResult(%arg0: tensor<4x!bmodelica.real>) -> tensor<?x?x!bmodelica.real> {
    %0 = bmodelica.diagonal %arg0 : tensor<4x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    return %0 : tensor<?x?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.diagonal %[[arg0]] : tensor<4x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicInputStaticResult
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x!bmodelica.real>)

func.func @DynamicInputStaticResult(%arg0: tensor<?x!bmodelica.real>) -> tensor<4x4x!bmodelica.real> {
    %0 = bmodelica.diagonal %arg0 : tensor<?x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    return %0 : tensor<4x4x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.diagonal %[[arg0]] : tensor<?x!bmodelica.real> -> tensor<4x4x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @DynamicInputDynamicResult
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x!bmodelica.real>)

func.func @DynamicInputDynamicResult(%arg0: tensor<?x!bmodelica.real>) -> tensor<?x?x!bmodelica.real> {
    %0 = bmodelica.diagonal %arg0 : tensor<?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    return %0 : tensor<?x?x!bmodelica.real>

    // CHECK: %[[result:.*]] = bmodelica.diagonal %[[arg0]] : tensor<?x!bmodelica.real> -> tensor<?x?x!bmodelica.real>
    // CHECK-NEXT: return %[[result]]
}
