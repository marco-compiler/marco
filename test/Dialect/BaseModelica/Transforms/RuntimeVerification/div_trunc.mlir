// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Integer operands

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.int>

    %0 = bmodelica.variable_get @a : !bmodelica.int

    // CHECK:      %[[rhs:.*]] = bmodelica.variable_get @b
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: integer division by zero"} {
    // CHECK-NEXT:     %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.neq %[[rhs]], %[[zero]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %1 = bmodelica.variable_get @b : !bmodelica.int
    %2 = bmodelica.div_trunc %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
}

// -----

// Real operands

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.real>

    %0 = bmodelica.variable_get @a : !bmodelica.real

    // CHECK:      %[[rhs:.*]] = bmodelica.variable_get @b
    // CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: integer division by zero"} {
    // CHECK-NEXT:     %[[epsilon:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:     %[[rhs_abs:.*]] = bmodelica.abs %[[rhs]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:     %[[cond:.*]] = bmodelica.gte %[[rhs_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:     bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT: }

    %1 = bmodelica.variable_get @b : !bmodelica.real
    %2 = bmodelica.div_trunc %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
}