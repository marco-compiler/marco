// RUN: modelica-opt %s --split-input-file --test-model-conversion | FileCheck %s

// CHECK: simulation.number_of_variables 2

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>
}
