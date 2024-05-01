// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK:   runtime.deinit_function {
// CHECK:       runtime.yield
// CHECK:   }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
        bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>
    }
}
