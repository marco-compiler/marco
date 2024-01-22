// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// Scalar variable.

// CHECK: simulation.printable_indices [true]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.real>
    }
}

// -----

// Array variable.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,2][0,1]}>
// CHECK: simulation.printable_indices [#[[index_set]]]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<3x2x!modelica.real>
    }
}
