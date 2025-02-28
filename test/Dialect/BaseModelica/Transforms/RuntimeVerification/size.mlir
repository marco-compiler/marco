// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// One-dimensional tensor

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @v : !bmodelica.variable<3x!bmodelica.real>
    %0 = bmodelica.variable_get @v : tensor<3x!bmodelica.real>

    // CHECK:      %[[index:.*]] = bmodelica.constant
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.size %0, %1 : (tensor<3x!bmodelica.real>, index) -> !bmodelica.int
}

// -----

// One-dimensional tensor + dimension index as function argument

// CHECK-LABEL: @test
// CHECK-SAME: %{{.*}}: tensor<3x!bmodelica.real>
// CHECK-SAME: %[[arg1:.*]]: index
func.func @test(%arg0: tensor<3x!bmodelica.real>, %arg1: index) -> !bmodelica.int {
    // CHECK:      bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.size %arg0, %arg1 : (tensor<3x!bmodelica.real>, index) -> !bmodelica.int
    return %0 : !bmodelica.int
}

// -----

// One-dimensional tensor, without dimension of interest

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @v : !bmodelica.variable<3x!bmodelica.real>

    // CHECK: %[[v:.*]] = bmodelica.variable_get @v : tensor<3x!bmodelica.real>
    // CHECK-NEXT: %{{[[:digit:]]+}} = bmodelica.size %[[v]] : tensor<3x!bmodelica.real> -> !bmodelica.int

    %0 = bmodelica.variable_get @v : tensor<3x!bmodelica.real>
    %1 = bmodelica.size %0 : tensor<3x!bmodelica.real> -> !bmodelica.int
}

// -----

// Multi-dimensional dynamic tensor

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @m : !bmodelica.variable<2x?x3x!bmodelica.real>
    %0 = bmodelica.variable_get @m : tensor<2x?x3x!bmodelica.real>

    // CHECK:      %[[index:.*]] = bmodelica.constant
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.size %0, %1 : (tensor<2x?x3x!bmodelica.real>, index) -> !bmodelica.int
}

// -----

// Multi-dimensional dynamic tensor + dimension index as function argument

// CHECK-LABEL: @test
// CHECK-SAME: %{{.*}}: tensor<2x?x3x!bmodelica.real>
// CHECK-SAME: %[[arg1:.*]]: index
func.func @test(%arg0: tensor<2x?x3x!bmodelica.real>, %arg1: index) -> !bmodelica.int {
    // CHECK:      bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:     %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:     %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %0 = bmodelica.size %arg0, %arg1 : (tensor<2x?x3x!bmodelica.real>, index) -> !bmodelica.int
    return %0 : !bmodelica.int
}

// -----

// Multi-dimensional dynamic tensor, without dimension of interest

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @v : !bmodelica.variable<2x?x3x!bmodelica.real>

    // CHECK: %[[v:.*]] = bmodelica.variable_get @v : tensor<2x?x3x!bmodelica.real>
    // CHECK-NEXT: %{{[[:digit:]]+}} = bmodelica.size %[[v]] : tensor<2x?x3x!bmodelica.real> -> !bmodelica.int

    %0 = bmodelica.variable_get @v : tensor<2x?x3x!bmodelica.real>
    %1 = bmodelica.size %0 : tensor<2x?x3x!bmodelica.real> -> !bmodelica.int
}
