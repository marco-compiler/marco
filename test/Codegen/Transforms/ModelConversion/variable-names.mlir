// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: name = "x"

// CHECK: #[[y:.*]] = #simulation.variable
// CHECK-SAME: name = "y"

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]], #[[y]]]

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    %1 = modelica.member_create @y : !modelica.member<3x!modelica.real>
    modelica.yield %0, %1 : !modelica.member<!modelica.real>, !modelica.member<3x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<3x!modelica.real>):

}
