// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for an array variable.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[el:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[array:.*]] = modelica.array_broadcast %[[el]]
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[array_load:.*]] = modelica.load %[[array]][%[[i0]]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[array_load]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.main_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.array_broadcast %0: !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %1 : !modelica.array<3x!modelica.int>
    }
}
