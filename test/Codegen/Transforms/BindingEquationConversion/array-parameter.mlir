// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for an array parameter.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[el:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[value:.*]] = modelica.array_broadcast %[[el]]
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.array_broadcast %0: !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %1 : !modelica.array<3x!modelica.int>
    }
}
