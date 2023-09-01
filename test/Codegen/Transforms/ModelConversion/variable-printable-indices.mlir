// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// Scalar variable.

// CHECK: simulation.printable_indices [true]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
}

// -----

// Array variable.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,2][0,1]}>
// CHECK: simulation.printable_indices [#[[index_set]]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x2x!modelica.real>
}
