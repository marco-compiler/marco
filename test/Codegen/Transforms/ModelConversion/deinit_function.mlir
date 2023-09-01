// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK:   simulation.deinit_function {
// CHECK:       simulation.yield
// CHECK:   }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>
}
