// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x?x3x!bmodelica.int>)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.int>) -> index {

    // CHECK-NOT:   bmodelica.assert
    // CHECK:       %{{.*}} = bmodelica.size %[[arg0]]

    %0 = bmodelica.size %arg0 : tensor<2x?x3x!bmodelica.int> -> index
    func.return %0 : index
}

// -----

// COM: Dimension specified

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<2x?x3x!bmodelica.int>, %[[arg1:.*]]: index)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.int>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg1]], %[[zero]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.size %arg0, %arg1 : (tensor<2x?x3x!bmodelica.int>, index) -> index
    func.return %0 : index
}