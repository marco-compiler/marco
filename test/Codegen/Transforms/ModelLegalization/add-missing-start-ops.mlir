// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Uninitialized scalar variable

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK-NEXT:  modelica.start (%[[x]] : !modelica.array<!modelica.int>) {each = false, fixed = false} {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):

}

// -----

// Uninitialized array variable

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>):
// CHECK-NEXT:  modelica.start (%[[x]] : !modelica.array<3x!modelica.int>) {each = true, fixed = false} {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int>
    modelica.yield %0 : !modelica.member<3x!modelica.int>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>):

}
