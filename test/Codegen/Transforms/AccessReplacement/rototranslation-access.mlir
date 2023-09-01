// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 3-d access.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<50x50x50x!modelica.real>
    modelica.variable @y : !modelica.variable<50x50x50x!modelica.real>
    modelica.variable @z : !modelica.variable<50x50x50x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]], %[[i2:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x : !modelica.array<50x50x50x!modelica.real>
    // CHECK-DAG:       %[[z:.*]] = modelica.variable_get @z : !modelica.array<50x50x50x!modelica.real>
    // CHECK-DAG:       %[[four:.*]] = modelica.constant 4 : index
    // CHECK-DAG:       %[[eight:.*]] = modelica.constant 8 : index
    // CHECK-DAG:       %[[minus_three:.*]] = modelica.constant -3 : index
    // CHECK-DAG:       %[[z_index_0:.*]] = modelica.add %[[i2]], %[[four]]
    // CHECK-DAG:       %[[z_index_1:.*]] = modelica.add %[[i0]], %[[eight]]
    // CHECK-DAG:       %[[z_index_2:.*]] = modelica.add %[[i1]], %[[minus_three]]
    // CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]], %[[i2]]]
    // CHECK-DAG:       %[[z_load:.*]] = modelica.load %[[z]][%[[z_index_0]], %[[z_index_1]], %[[z_index_2]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load::*]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[z_load:.*]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i][j][z] = y[j - 7][z + 5][i + 3]
    %t0 = modelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<50x50x50x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<50x50x50x!modelica.real>
        %2 = modelica.constant 7 : index
        %3 = modelica.constant 5 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.sub %i1, %2 : (index, index) -> index
        %6 = modelica.add %i2, %3 : (index, index) -> index
        %7 = modelica.add %i0, %4 : (index, index) -> index
        %8 = modelica.load %0[%i0, %i1, %i2] : !modelica.array<50x50x50x!modelica.real>
        %9 = modelica.load %1[%5, %6, %7] : !modelica.array<50x50x50x!modelica.real>
        %10 = modelica.equation_side %8 : tuple<!modelica.real>
        %11 = modelica.equation_side %9 : tuple<!modelica.real>
        modelica.equation_sides %10, %11 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [10,20][10,20][10,20]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [10,20][10,20][10,20]>, replace_indices = #modeling<index_set {[10,20][10,20][10,20]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[z - 4][i + 1][j - 5] = z[i][j][z]
    %t1 = modelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<50x50x50x!modelica.real>
        %1 = modelica.variable_get @z : !modelica.array<50x50x50x!modelica.real>
        %2 = modelica.constant 4 : index
        %3 = modelica.constant 1 : index
        %4 = modelica.constant 5 : index
        %5 = modelica.sub %i2, %2 : (index, index) -> index
        %6 = modelica.add %i0, %3 : (index, index) -> index
        %7 = modelica.sub %i1, %4 : (index, index) -> index
        %8 = modelica.load %0[%5, %6, %7] : !modelica.array<50x50x50x!modelica.real>
        %9 = modelica.load %1[%i0, %i1, %i2] : !modelica.array<50x50x50x!modelica.real>
        %10 = modelica.equation_side %8 : tuple<!modelica.real>
        %11 = modelica.equation_side %9 : tuple<!modelica.real>
        modelica.equation_sides %10, %11 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [10,20][10,20][10,20]>} : !modelica.equation
}
