// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 2-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<50x50x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<50x50x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<50x50x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : tensor<50x50x!bmodelica.real>
    // CHECK-DAG:       %[[z:.*]] = bmodelica.variable_get @z : tensor<50x50x!bmodelica.real>
    // CHECK-DAG:       %[[x_load:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[z_load:.*]] = bmodelica.tensor_extract %[[z]][%[[i1]], %[[i0]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[z_load]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = z[j, i]
     %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<50x50x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<50x50x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1] : tensor<50x50x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[j][i] = z[i][j]
    %t1 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @z : tensor<50x50x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i1, %i0] : tensor<50x50x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1] : tensor<50x50x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][0,2]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][0,2]>, replace_indices = #modeling<multidim_range [0,2][0,2]>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2][0,2]>}
    }
}

// -----

// 3-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<50x50x50x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<50x50x50x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<50x50x50x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]], %[[i2:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : tensor<50x50x50x!bmodelica.real>
    // CHECK-DAG:       %[[z:.*]] = bmodelica.variable_get @z : tensor<50x50x50x!bmodelica.real>
    // CHECK-DAG:       %[[x_load:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]], %[[i2]]]
    // CHECK-DAG:       %[[z_load:.*]] = bmodelica.tensor_extract %[[z]][%[[i2]], %[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[z_load]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]] : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    // CHECK-NEXT:  }

    // x[i][j][z] = y[j][z][i]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<50x50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<50x50x50x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0, %i1, %i2] : tensor<50x50x50x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i1, %i2, %i0] : tensor<50x50x50x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[z][i][j] = z[i][j][z]
    %t1 = bmodelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<50x50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @z : tensor<50x50x50x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i2, %i0, %i1] : tensor<50x50x50x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0, %i1, %i2] : tensor<50x50x50x!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][0,2][0,2]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][0,2][0,2]>, replace_indices = #modeling<multidim_range [0,2][0,2][0,2]>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2][0,2][0,2]>}
    }
}
