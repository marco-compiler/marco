// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// One-dimensional array

// CHECK-LABEL: @test
func.func @test(%arg0: !bmodelica.array<3x!bmodelica.real>) -> index {
    // CHECK:      %[[index:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.constant 1 : index
    %1 = bmodelica.dim %arg0, %0 : !bmodelica.array<3x!bmodelica.real>
    return %1 : index
}

// -----

// One-dimensional array + dimension index as function argument

// CHECK-LABEL: @test
// CHECK-SAME: %{{.*}}: !bmodelica.array<3x!bmodelica.real>
// CHECK-SAME: %[[arg1:.*]]: index
func.func @test(%arg0: !bmodelica.array<3x!bmodelica.real>, %arg1: index) -> index {
    // CHECK:      bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<3x!bmodelica.real>
    return %0 : index
}

// -----

// Multi-dimensional dynamic array

// CHECK-LABEL: @test
func.func @test(%arg0: !bmodelica.array<2x?x3x!bmodelica.real>, %arg1: index) -> index {
    // CHECK:      %[[index:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.constant 1 : index
    %1 = bmodelica.dim %arg0, %0 : !bmodelica.array<2x?x3x!bmodelica.real>
    return %1 : index
}

// -----

// Multi-dimensional dynamic array + dimension index as function argument

// CHECK-LABEL: @test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x?x3x!bmodelica.real>
// CHECK-SAME: %[[arg1:.*]]: index
func.func @test(%arg0: !bmodelica.array<2x?x3x!bmodelica.real>, %arg1: index) -> index {
    // CHECK:      bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<2x?x3x!bmodelica.real>
    return %0 : index
}
