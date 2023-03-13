// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK:       simulation.init_function () -> (!modelica.array<!modelica.real>, !modelica.array<3x!modelica.real>) {
// CHECK-NEXT:      %[[x:.*]] = modelica.alloc : !modelica.array<!modelica.real>
// CHECK-NEXT:      %[[xValue:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      modelica.store %[[x]][], %[[xValue]]
// CHECK-NEXT:      %[[y:.*]] = modelica.alloc : !modelica.array<3x!modelica.real>
// CHECK-NEXT:      %[[yValue:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      modelica.array_fill %[[y]], %[[yValue]]
// CHECK-NEXT:      simulation.yield %[[x]], %[[y]] : !modelica.array<!modelica.real>, !modelica.array<3x!modelica.real>
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>
}
