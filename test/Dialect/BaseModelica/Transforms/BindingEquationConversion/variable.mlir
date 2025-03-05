// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// COM: Binding equation for a scalar variable.

// CHECK-LABEL: @scalarVariable

bmodelica.model @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template
    // CHECK-DAG:   %[[lhsValue:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:   %[[rhsValue:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
    // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
    // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK: bmodelica.dynamic
    // CHECK: bmodelica.equation_instance %[[t0]]
}

// -----

// COM: Binding equation for an array variable.

// CHECK-LABEL: @arrayVariable

bmodelica.model @arrayVariable {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      %1 = bmodelica.tensor_broadcast %0: !bmodelica.int -> tensor<3x!bmodelica.int>
      bmodelica.yield %1 : tensor<3x!bmodelica.int>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:   %[[el:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:   %[[tensor:.*]] = bmodelica.tensor_broadcast %[[el]]
    // CHECK-DAG:   %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]]]
    // CHECK-DAG:   %[[tensor_extract:.*]] = bmodelica.tensor_extract %[[tensor]][%[[i0]]]
    // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
    // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[tensor_extract]]
    // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK:       bmodelica.dynamic
    // CHECK:       bmodelica.equation_instance %[[t0]]
    // CHECK-SAME:  indices = {[0,2]}
}
