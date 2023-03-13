// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: name = "x"

// CHECK: #[[y:.*]] = #simulation.variable
// CHECK-SAME: name = "y"

// CHECK: simulation.module
// CHECK-SAME: variables = [#[[x]], #[[y]]]

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>
    modelica.variable @y : !modelica.member<3x!modelica.real>
}
