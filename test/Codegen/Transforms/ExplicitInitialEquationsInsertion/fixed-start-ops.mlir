// RUN: modelica-opt %s --split-input-file --insert-explicit-initial-equations | FileCheck %s

// Scalar variable with fixed start value.

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.initial_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = false, fixed = true}
}

// -----

// Array variable with fixed start scalar value.

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.initial_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = true, fixed = true}
}

// -----

// Array variable with fixed start array value

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[value_0:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[value_1:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[value_2:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG:       %[[array:.*]] = modelica.array_from_elements %[[value_0]], %[[value_1]], %[[value_2]]
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[array_load:.*]] = modelica.load %[[array]][%[[i0]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[array_load]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.initial_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        %1 = modelica.constant #modelica.int<1> : !modelica.int
        %2 = modelica.constant #modelica.int<2> : !modelica.int
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    } {each = false, fixed = true}
}
