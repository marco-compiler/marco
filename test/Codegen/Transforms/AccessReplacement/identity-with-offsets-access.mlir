// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 1-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<50x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<50x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<50x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[z:.*]] = bmodelica.variable_get @z
    // CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[three:.*]] = bmodelica.constant 3 : index
    // CHECK-DAG:       %[[z_index:.*]] = bmodelica.add %[[i0]], %[[three]]
    // CHECK-DAG:       %[[z_load:.*]] = bmodelica.load %[[z]][%[[z_index]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[z_load]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i] = y[i + 2]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<50x!bmodelica.real>
        %1 = bmodelica.variable_get @y : !bmodelica.array<50x!bmodelica.real>
        %2 = bmodelica.load %0[%i0] : !bmodelica.array<50x!bmodelica.real>
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.add %i0, %3 : (index, index) -> index
        %5 = bmodelica.load %1[%4] : !bmodelica.array<50x!bmodelica.real>
        %6 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[i - 1] = z[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<50x!bmodelica.real>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.sub %i0, %1 : (index, index) -> index
        %3 = bmodelica.load %0[%2] : !bmodelica.array<50x!bmodelica.real>
        %4 = bmodelica.variable_get @z : !bmodelica.array<50x!bmodelica.real>
        %5 = bmodelica.load %4[%i0] : !bmodelica.array<50x!bmodelica.real>
        %6 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5: tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2]>, replace_indices = #modeling<index_set {[0,2]}>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [1,4]>} : !bmodelica.equation
    }
}

// -----

// 2-d access.

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<50x50x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<50x50x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<50x50x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.array<50x50x!bmodelica.real>
    // CHECK-DAG:       %[[z:.*]] = bmodelica.variable_get @z : !bmodelica.array<50x50x!bmodelica.real>
    // CHECK-DAG:       %[[three:.*]] = bmodelica.constant 3 : index
    // CHECK-DAG:       %[[minus_ten:.*]] = bmodelica.constant -10 : index
    // CHECK-DAG:       %[[z_index_0:.*]] = bmodelica.add %[[i0]], %[[three]]
    // CHECK-DAG:       %[[z_index_1:.*]] = bmodelica.add %[[i1]], %[[minus_ten]]
    // CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[z_load:.*]] = bmodelica.load %[[z]][%[[z_index_0]], %[[z_index_1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[z_load]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i][j] = y[i + 2][j - 6]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @y : !bmodelica.array<50x50x!bmodelica.real>
        %2 = bmodelica.load %0[%i0, %i1] : !bmodelica.array<50x50x!bmodelica.real>
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 6 : index
        %5 = bmodelica.add %i0, %3 : (index, index) -> index
        %6 = bmodelica.sub %i1, %4 : (index, index) -> index
        %7 = bmodelica.load %1[%5, %6] : !bmodelica.array<50x50x!bmodelica.real>
        %8 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[i - 1][j + 4] = z[i][j]
    %t1 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<50x50x!bmodelica.real>
        %1 = bmodelica.variable_get @z : !bmodelica.array<50x50x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 4 : index
        %4 = bmodelica.sub %i0, %2 : (index, index) -> index
        %5 = bmodelica.add %i1, %3 : (index, index) -> index
        %6 = bmodelica.load %0[%4, %5] : !bmodelica.array<50x50x!bmodelica.real>
        %7 = bmodelica.load %1[%i0, %i1] : !bmodelica.array<50x50x!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7: tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][10,13]>}
        bmodelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][10,13]>, replace_indices = #modeling<index_set {[0,2][10,13]}>, replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation

        bmodelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [1,10][0,10]>} : !bmodelica.equation
    }
}
