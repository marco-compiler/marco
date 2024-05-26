// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for a scalar variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] {
// CHECK-DAG:       %[[lhsValue:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[rhsValue:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    }
}
