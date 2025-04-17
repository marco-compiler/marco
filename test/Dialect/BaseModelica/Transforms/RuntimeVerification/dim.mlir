// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.array<2x?x3x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @Test(%arg0: !bmodelica.array<2x?x3x!bmodelica.real>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg1]], %[[zero]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<2x?x3x!bmodelica.real>
    func.return %0 : index
}
