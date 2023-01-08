// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK:   simulation.deinit_function(%[[x:.*]]: !modelica.array<!modelica.real>, %[[y:.*]]: !modelica.array<3x!modelica.real>) {
// CHECK:       modelica.free %[[x]]
// CHECK:       modelica.free %[[y]]
// CHECK:       simulation.yield
// CHECK:   }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    %1 = modelica.member_create @y : !modelica.member<3x!modelica.real>
    modelica.yield %0, %1 : !modelica.member<!modelica.real>, !modelica.member<3x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<3x!modelica.real>):

}
