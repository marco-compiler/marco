// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK: simulation.number_of_variables 2

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @y : !modelica.variable<3x!modelica.real>
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
    modelica.simulation_variable @y : !modelica.variable<3x!modelica.int>
}
