// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for an array variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[el:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[array:.*]] = bmodelica.array_broadcast %[[el]]
// CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[array_load:.*]] = bmodelica.load %[[array]][%[[i0]]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[array_load]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
      %1 = bmodelica.array_broadcast %0: !bmodelica.int -> !bmodelica.array<3x!bmodelica.int>
      bmodelica.yield %1 : !bmodelica.array<3x!bmodelica.int>
    }
}
