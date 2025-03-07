// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s


// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x!bmodelica.int>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.variable_get @arg : !bmodelica.array<2x!bmodelica.int>

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %1 : <2x!bmodelica.int> -> tensor<2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %[[conv]], %[[constzero]] : (tensor<2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %3 = bmodelica.subscription %1[%0] : <2x!bmodelica.int>, index -> !bmodelica.array<!bmodelica.real>
}

// -----

// Multidimensional

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x2x!bmodelica.int>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.variable_get @arg : !bmodelica.array<2x2x!bmodelica.int>

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %2 : <2x2x!bmodelica.int> -> tensor<2x2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %[[conv]], %[[constzero]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %[[conv]], %[[constone]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %4 = bmodelica.subscription %2[%0, %1] : <2x2x!bmodelica.int>, index, index -> !bmodelica.array<!bmodelica.int>
}