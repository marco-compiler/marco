// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Dynamic tensor

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x?x3x!bmodelica.real>)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.real>) -> index {

    // CHECK: %{{.*}} = bmodelica.size %[[arg0]] : tensor<2x?x3x!bmodelica.real> -> index

    %0 = bmodelica.size %arg0 : tensor<2x?x3x!bmodelica.real> -> index
    func.return %0 : index
}

// -----

// COM: Dynamic tensor with specified dimension

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<2x?x3x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.real>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.size %arg0, %arg1 : (tensor<2x?x3x!bmodelica.real>, index) -> index
    func.return %0 : index
}