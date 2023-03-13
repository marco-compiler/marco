// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-NOT: dimensions

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
}

// -----

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: dimensions = [3]

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.real>
}

// -----

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: dimensions = [3, 2]

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x2x!modelica.real>
}
