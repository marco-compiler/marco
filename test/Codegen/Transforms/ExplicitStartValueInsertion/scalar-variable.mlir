// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// Uninitialized scalar variable.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
}
