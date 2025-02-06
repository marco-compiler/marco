// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 1-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i] = y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<3x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3: tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[i] = 0
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2: tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2]>, replace_indices = #modeling<multidim_range [0,2]>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2]>}
    }
}

// -----

// 2-d access.

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

    // y[i, j] = 0
    %t1 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x4x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<3x4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2: tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][0,3]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][0,3]>, replace_indices = #modeling<multidim_range [0,2][0,3]>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2][0,3]>}
    }
}
