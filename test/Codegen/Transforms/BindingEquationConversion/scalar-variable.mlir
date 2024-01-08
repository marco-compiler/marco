// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for a scalar variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-DAG:       %[[lhsValue:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[rhsValue:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.main_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}
