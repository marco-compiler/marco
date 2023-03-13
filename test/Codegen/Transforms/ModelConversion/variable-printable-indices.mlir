// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// Scalar variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-NOT: printable_indices

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>
}

// -----

// Array variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: printable_indices = [<[0, 3], [0, 2]>]

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x2x!modelica.real>
}
