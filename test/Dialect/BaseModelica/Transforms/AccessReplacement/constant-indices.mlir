// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x4x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x4x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3: tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[2, 3] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.constant 2 : index
        %2 = bmodelica.constant 3 : index
        %3 = bmodelica.tensor_extract %0[%1, %2] : tensor<3x4x!bmodelica.real>
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4: tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [2,2][3,3]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [2,2][3,3]>, replace_indices = #modeling<index_set {[2,2][3,3]}>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation

        bmodelica.equation_instance %t1 {id = "eq1"} : !bmodelica.equation
    }
}

// -----

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x4x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x4x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[i0]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3: tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[i, 3] = i
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.tensor_extract %0[%i0, %1] : tensor<3x4x!bmodelica.real>
        %3 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %i0: tuple<index>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<index>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][3,3]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][3,3]>, replace_indices = #modeling<index_set {[0,2][3,3]}>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,3]>} : !bmodelica.equation
    }
}

// -----

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x4x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x4x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[i1]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3: tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[2, i] = i
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.constant 2 : index
        %2 = bmodelica.tensor_extract %0[%1, %i0] : tensor<3x4x!bmodelica.real>
        %3 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %i0: tuple<index>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<index>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [2,2][0,3]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [2,2][0,3]>, replace_indices = #modeling<index_set {[2,2][0,3]}>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,3]>} : !bmodelica.equation
    }
}
