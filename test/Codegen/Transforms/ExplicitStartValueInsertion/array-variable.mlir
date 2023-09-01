// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// Uninitialized array variable.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[value]]
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
}
