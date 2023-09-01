// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for an array variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[el:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[el]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[array]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.equation_instance %[[t0]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.array_broadcast %0: !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %1 : !modelica.array<3x!modelica.int>
    }
}
