// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.array<3x?x?x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @Test(%arg0: !bmodelica.array<3x?x?x!bmodelica.real>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<3x?x?x!bmodelica.real>
    func.return %0 : index
}
