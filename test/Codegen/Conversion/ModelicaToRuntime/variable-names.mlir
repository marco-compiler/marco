// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK: runtime.variable_names ["x", "y"]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @y : !modelica.variable<3x!modelica.real>
    }
}
