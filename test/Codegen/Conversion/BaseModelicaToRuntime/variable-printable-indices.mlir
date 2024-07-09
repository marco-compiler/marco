// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime | FileCheck %s

// Scalar variable.

// CHECK: runtime.printable_indices [true]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    }
}

// -----

// Array variable.

// CHECK: runtime.printable_indices [{[0,2][0,1]}]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.real>
    }
}
