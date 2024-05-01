// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// Scalar variable.

// CHECK: runtime.printable_indices [true]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    }
}

// -----

// Array variable.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,2][0,1]}>
// CHECK: runtime.printable_indices [#[[index_set]]]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.real>
    }
}
