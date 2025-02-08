// RUN: modelica-opt %s --split-input-file --insert-explicit-initial-equations | FileCheck %s

// COM: Scalar variable with fixed start value.

// CHECK-LABEL: @scalarFixedStart

bmodelica.model @scalarFixedStart {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
        bmodelica.yield %0 : !bmodelica.int
    } {each = false, fixed = true}

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = []
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[value]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK: bmodelica.initial
    // CHECK: bmodelica.equation_instance %[[t0]]
}

// -----

// COM: Array variable with fixed start scalar value.

// CHECK-LABEL: @arrayFixedScalarStart

bmodelica.model @arrayFixedScalarStart {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
        bmodelica.yield %0 : !bmodelica.int
    } {each = true, fixed = true}

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[value]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK:       bmodelica.initial
    // CHECK:       bmodelica.equation_instance %[[t0]]
    // CHECK-SAME:  indices = #modeling<multidim_range [0,2]>
}

// -----

// COM: Array variable with fixed start array value

// CHECK-LABEL: @arrayFixedArrayStart

bmodelica.model @arrayFixedArrayStart {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
        %2 = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
        %3 = bmodelica.tensor_from_elements %0, %1, %2 : !bmodelica.int, !bmodelica.int, !bmodelica.int -> tensor<3x!bmodelica.int>
        bmodelica.yield %3 : tensor<3x!bmodelica.int>
    } {each = false, fixed = true}

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[value_0:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:       %[[value_1:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK-DAG:       %[[value_2:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK-DAG:       %[[tensor:.*]] = bmodelica.tensor_from_elements %[[value_0]], %[[value_1]], %[[value_2]]
    // CHECK-DAG:       %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[tensor_extract:.*]] = bmodelica.tensor_extract %[[tensor]][%[[i0]]]
    // CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
    // CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[tensor_extract]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK:       bmodelica.initial
    // CHECK:       bmodelica.equation_instance %[[t0]]
    // CHECK-SAME:  indices = #modeling<multidim_range [0,2]>
}
