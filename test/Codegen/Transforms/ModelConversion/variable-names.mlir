// RUN: modelica-opt %s --split-input-file --test-model-conversion | FileCheck %s

// CHECK: simulation.variable_names ["x", "y"]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>
}
