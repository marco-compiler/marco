// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK: runtime.variable_names ["x", "y"]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
        bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>
    }
}
