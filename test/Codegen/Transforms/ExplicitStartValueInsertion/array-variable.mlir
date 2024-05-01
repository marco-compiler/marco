// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// Uninitialized array variable.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
// CHECK-NEXT:      %[[array:.*]] = bmodelica.array_broadcast %[[value]]
// CHECK-NEXT:      bmodelica.yield %[[array]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
}
