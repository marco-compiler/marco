// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Integer operand

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of acos outside the domain. It should be -1 <= arg <= 1"} {
    // CHECK-NEXT:    %[[lower_bound:.*]] = bmodelica.constant #bmodelica<int -1> : !bmodelica.int
    // CHECK-NEXT:    %[[upper_bound:.*]] = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
    // CHECK-NEXT:    %[[cond1:.*]] = bmodelica.gte %[[arg]], %[[lower_bound]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[cond2:.*]] = bmodelica.lte %[[arg]], %[[upper_bound]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.variable_get @x: !bmodelica.int
    %1 = bmodelica.acos %0 : !bmodelica.int -> !bmodelica.real
}

// -----

// Real operand

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of acos outside the domain. It should be -1 <= arg <= 1"} {
    // CHECK-NEXT:      %[[lower_bound:.*]] = bmodelica.constant #bmodelica<real -1.000000e+00> : !bmodelica.real
    // CHECK-NEXT:      %[[upper_bound:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00> : !bmodelica.real
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[arg]], %[[lower_bound]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lte %[[arg]], %[[upper_bound]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.variable_get @x: !bmodelica.real
    %1 = bmodelica.acos %0 : !bmodelica.real -> !bmodelica.real
}