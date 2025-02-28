// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Integer operand

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of log outside the domain. It should be > 0"} {
    // CHECK-NEXT:    %[[constant:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.gt %[[arg]], %[[constant]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.variable_get @x : !bmodelica.int
    %1 = bmodelica.log %0 : !bmodelica.int -> !bmodelica.real
}

// -----

// Real operand

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of log outside the domain. It should be > 0"} {
    // CHECK-NEXT:    %[[constant:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.gt %[[arg]], %[[constant]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.variable_get @x : !bmodelica.real
    %1 = bmodelica.log %0 : !bmodelica.real -> !bmodelica.real
}
