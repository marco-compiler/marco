// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for a scalar parameter.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    }
}
