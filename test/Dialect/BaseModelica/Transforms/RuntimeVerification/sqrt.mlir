// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operand

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x : !bmodelica.int
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of sqrt outside the domain. It should be >= 0"} {
    // CHECK-NEXT:    %[[constant:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.gte %[[arg]], %[[constant]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    // CHECK-NEXT:  %{{[0-9]+}} = bmodelica.sqrt %[[arg]] : !bmodelica.int -> !bmodelica.real

    %0 = bmodelica.variable_get @x : !bmodelica.int
    %1 = bmodelica.sqrt %0 : !bmodelica.int -> !bmodelica.real
}

// -----

// COM: Real operand

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    // CHECK:       %[[arg:.*]] = bmodelica.variable_get @x : !bmodelica.real
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: Argument of sqrt outside the domain. It should be >= 0"} {
    // CHECK-NEXT:    %[[constant:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.gte %[[arg]], %[[constant]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.variable_get @x : !bmodelica.real
    %1 = bmodelica.sqrt %0 : !bmodelica.real -> !bmodelica.real
}
