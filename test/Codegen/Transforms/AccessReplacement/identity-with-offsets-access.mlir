// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 1-d access.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<50x!modelica.real>
    modelica.variable @y : !modelica.variable<50x!modelica.real>
    modelica.variable @z : !modelica.variable<50x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[z:.*]] = modelica.variable_get @z
    // CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[three:.*]] = modelica.constant 3 : index
    // CHECK-DAG:       %[[z_index:.*]] = modelica.add %[[i0]], %[[three]]
    // CHECK-DAG:       %[[z_load:.*]] = modelica.load %[[z]][%[[z_index]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[z_load]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i] = y[i + 2]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<50x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<50x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<50x!modelica.real>
        %3 = modelica.constant 2 : index
        %4 = modelica.add %i0, %3 : (index, index) -> index
        %5 = modelica.load %1[%4] : !modelica.array<50x!modelica.real>
        %6 = modelica.equation_side %2 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2]>, replace_indices = #modeling<index_set {[0,2]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[i - 1] = z[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<50x!modelica.real>
        %1 = modelica.constant 1 : index
        %2 = modelica.sub %i0, %1 : (index, index) -> index
        %3 = modelica.load %0[%2] : !modelica.array<50x!modelica.real>
        %4 = modelica.variable_get @z : !modelica.array<50x!modelica.real>
        %5 = modelica.load %4[%i0] : !modelica.array<50x!modelica.real>
        %6 = modelica.equation_side %3 : tuple<!modelica.real>
        %7 = modelica.equation_side %5: tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [1,4]>} : !modelica.equation
}

// -----

// 2-d access.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<50x50x!modelica.real>
    modelica.variable @y : !modelica.variable<50x50x!modelica.real>
    modelica.variable @z : !modelica.variable<50x50x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x : !modelica.array<50x50x!modelica.real>
    // CHECK-DAG:       %[[z:.*]] = modelica.variable_get @z : !modelica.array<50x50x!modelica.real>
    // CHECK-DAG:       %[[three:.*]] = modelica.constant 3 : index
    // CHECK-DAG:       %[[minus_ten:.*]] = modelica.constant -10 : index
    // CHECK-DAG:       %[[z_index_0:.*]] = modelica.add %[[i0]], %[[three]]
    // CHECK-DAG:       %[[z_index_1:.*]] = modelica.add %[[i1]], %[[minus_ten]]
    // CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[z_load:.*]] = modelica.load %[[z]][%[[z_index_0]], %[[z_index_1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[z_load]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i][j] = y[i + 2][j - 6]
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<50x50x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<50x50x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<50x50x!modelica.real>
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 6 : index
        %5 = modelica.add %i0, %3 : (index, index) -> index
        %6 = modelica.sub %i1, %4 : (index, index) -> index
        %7 = modelica.load %1[%5, %6] : !modelica.array<50x50x!modelica.real>
        %8 = modelica.equation_side %2 : tuple<!modelica.real>
        %9 = modelica.equation_side %7 : tuple<!modelica.real>
        modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][10,13]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][10,13]>, replace_indices = #modeling<index_set {[0,2][10,13]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[i - 1][j + 4] = z[i][j]
    %t1 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<50x50x!modelica.real>
        %1 = modelica.variable_get @z : !modelica.array<50x50x!modelica.real>
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 4 : index
        %4 = modelica.sub %i0, %2 : (index, index) -> index
        %5 = modelica.add %i1, %3 : (index, index) -> index
        %6 = modelica.load %0[%4, %5] : !modelica.array<50x50x!modelica.real>
        %7 = modelica.load %1[%i0, %i1] : !modelica.array<50x50x!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        %9 = modelica.equation_side %7: tuple<!modelica.real>
        modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [1,10][0,10]>} : !modelica.equation
}
