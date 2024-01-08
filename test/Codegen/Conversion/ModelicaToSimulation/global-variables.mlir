// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK-DAG: modelica.global_variable @[[x:.*]] : !modelica.array<!modelica.int>
// CHECK-DAG: modelica.global_variable @[[y:.*]] : !modelica.array<3x!modelica.int>

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
        modelica.variable @y : !modelica.variable<3x!modelica.int>
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
    modelica.simulation_variable @y : !modelica.variable<3x!modelica.int>
}
