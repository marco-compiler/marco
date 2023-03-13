// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Uninitialized scalar variable.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>
}

// -----

// Uninitialized array variable.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = true, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>
}
