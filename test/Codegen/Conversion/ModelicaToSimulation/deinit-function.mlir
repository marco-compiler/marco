// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK:   simulation.deinit_function {
// CHECK:       simulation.yield
// CHECK:   }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @y : !modelica.variable<3x!modelica.real>
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.real>
    modelica.simulation_variable @y : !modelica.variable<3x!modelica.real>
}
