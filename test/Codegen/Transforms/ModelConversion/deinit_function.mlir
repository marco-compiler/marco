// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK:   simulation.deinit_function(%[[x:.*]]: !modelica.array<!modelica.real>, %[[y:.*]]: !modelica.array<3x!modelica.real>) {
// CHECK:       modelica.free %[[x]]
// CHECK:       modelica.free %[[y]]
// CHECK:       simulation.yield
// CHECK:   }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>
    modelica.variable @y : !modelica.member<3x!modelica.real>
}
