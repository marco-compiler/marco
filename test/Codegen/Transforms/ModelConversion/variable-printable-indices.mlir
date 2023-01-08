// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// Scalar variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-NOT: printable_indices

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    modelica.yield %0 : !modelica.member<!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>):

}

// -----

// Array variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: printable_indices = [<[0, 3], [0, 2]>]

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x2x!modelica.real>
    modelica.yield %0 : !modelica.member<3x2x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<3x2x!modelica.real>):

}
