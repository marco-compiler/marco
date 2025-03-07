// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Mono-dimensional

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x!bmodelica.int>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.variable_get @arg : !bmodelica.array<2x!bmodelica.int>

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %1 : <2x!bmodelica.int> -> tensor<2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[indexzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %[[conv]], %[[indexzero]] : (tensor<2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[constzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %2 = bmodelica.load %1[%0] : !bmodelica.array<2x!bmodelica.int>

}

// -----

// Multi-dimensional

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x2x!bmodelica.int>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.variable_get @arg : !bmodelica.array<2x2x!bmodelica.int>

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %2 : <2x2x!bmodelica.int> -> tensor<2x2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[indexzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %[[conv]], %[[indexzero]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[constzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:  %[[constzero_2:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[indexzero_2:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[size_2:.*]] = bmodelica.size %[[conv]], %[[indexzero_2]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte_2:.*]] = bmodelica.gte %1, %[[constzero_2]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt_2:.*]] = bmodelica.lt %1, %[[size_2]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond_2:.*]] = bmodelica.and %[[gte_2]], %[[lt_2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond_2]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %3 = bmodelica.load %2[%0, %1] : !bmodelica.array<2x2x!bmodelica.int>

}