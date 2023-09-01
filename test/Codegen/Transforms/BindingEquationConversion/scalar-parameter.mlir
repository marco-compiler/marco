// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for a scalar parameter.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int, parameter>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}
