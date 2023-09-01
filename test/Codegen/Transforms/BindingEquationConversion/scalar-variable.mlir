// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for a scalar variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.equation_instance %[[t0]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}
