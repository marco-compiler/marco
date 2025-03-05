// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 3-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<50x50x50x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<50x50x50x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<50x50x50x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]], %[[i2:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : tensor<50x50x50x!bmodelica.real>
    // CHECK-DAG:       %[[z:.*]] = bmodelica.variable_get @z : tensor<50x50x50x!bmodelica.real>
    // CHECK-DAG:       %[[four:.*]] = bmodelica.constant 4 : index
    // CHECK-DAG:       %[[eight:.*]] = bmodelica.constant 8 : index
    // CHECK-DAG:       %[[minus_three:.*]] = bmodelica.constant -3 : index
    // CHECK-DAG:       %[[z_index_0:.*]] = bmodelica.add %[[i2]], %[[four]]
    // CHECK-DAG:       %[[z_index_1:.*]] = bmodelica.add %[[i0]], %[[eight]]
    // CHECK-DAG:       %[[z_index_2:.*]] = bmodelica.add %[[i1]], %[[minus_three]]
    // CHECK-DAG:       %[[x_load:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]], %[[i2]]]
    // CHECK-DAG:       %[[z_load:.*]] = bmodelica.tensor_extract %[[z]][%[[z_index_0]], %[[z_index_1]], %[[z_index_2]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load::*]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[z_load:.*]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i][j][z] = y[j - 7][z + 5][i + 3]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<50x50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<50x50x50x!bmodelica.real>
        %2 = bmodelica.constant 7 : index
        %3 = bmodelica.constant 5 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.sub %i1, %2 : (index, index) -> index
        %6 = bmodelica.add %i2, %3 : (index, index) -> index
        %7 = bmodelica.add %i0, %4 : (index, index) -> index
        %8 = bmodelica.tensor_extract %0[%i0, %i1, %i2] : tensor<50x50x50x!bmodelica.real>
        %9 = bmodelica.tensor_extract %1[%5, %6, %7] : tensor<50x50x50x!bmodelica.real>
        %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
        %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
        bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[z - 4][i + 1][j - 5] = z[i][j][z]
    %t1 = bmodelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<50x50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @z : tensor<50x50x50x!bmodelica.real>
        %2 = bmodelica.constant 4 : index
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.constant 5 : index
        %5 = bmodelica.sub %i2, %2 : (index, index) -> index
        %6 = bmodelica.add %i0, %3 : (index, index) -> index
        %7 = bmodelica.sub %i1, %4 : (index, index) -> index
        %8 = bmodelica.tensor_extract %0[%5, %6, %7] : tensor<50x50x50x!bmodelica.real>
        %9 = bmodelica.tensor_extract %1[%i0, %i1, %i2] : tensor<50x50x50x!bmodelica.real>
        %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
        %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
        bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]], indices = {[10,20][10,20][10,20]} {id = "eq0"}
        bmodelica.equation_instance %t0, indices = {[10,20][10,20][10,20]} {id = "eq0", replace_indices = #modeling<multidim_range [10,20][10,20][10,20]>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1, indices = {[10,20][10,20][10,20]} {id = "eq1"}
    }
}
