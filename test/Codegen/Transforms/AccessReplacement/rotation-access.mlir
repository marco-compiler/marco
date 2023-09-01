// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// 2-d access.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<50x50x!modelica.real>
    modelica.variable @y : !modelica.variable<50x50x!modelica.real>
    modelica.variable @z : !modelica.variable<50x50x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x : !modelica.array<50x50x!modelica.real>
    // CHECK-DAG:       %[[z:.*]] = modelica.variable_get @z : !modelica.array<50x50x!modelica.real>
    // CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[z_load:.*]] = modelica.load %[[z]][%[[i1]], %[[i0]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[z_load]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = z[j, i]
     %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<50x50x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<50x50x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<50x50x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<50x50x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][0,2]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][0,2]>, replace_indices = #modeling<index_set {[0,2][0,2]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[j][i] = z[i][j]
    %t1 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<50x50x!modelica.real>
        %1 = modelica.variable_get @z : !modelica.array<50x50x!modelica.real>
        %2 = modelica.load %0[%i1, %i0] : !modelica.array<50x50x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<50x50x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2][0,2]>} : !modelica.equation
}

// -----

// 3-d access.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<50x50x50x!modelica.real>
    modelica.variable @y : !modelica.variable<50x50x50x!modelica.real>
    modelica.variable @z : !modelica.variable<50x50x50x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]], %[[i2:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x : !modelica.array<50x50x50x!modelica.real>
    // CHECK-DAG:       %[[z:.*]] = modelica.variable_get @z : !modelica.array<50x50x50x!modelica.real>
    // CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]], %[[i2]]]
    // CHECK-DAG:       %[[z_load:.*]] = modelica.load %[[z]][%[[i2]], %[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[z_load]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]] : tuple<!modelica.real>, tuple<!modelica.real>
    // CHECK-NEXT:  }

    // x[i][j][z] = y[j][z][i]
    %t0 = modelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<50x50x50x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<50x50x50x!modelica.real>
        %2 = modelica.load %0[%i0, %i1, %i2] : !modelica.array<50x50x50x!modelica.real>
        %3 = modelica.load %1[%i1, %i2, %i0] : !modelica.array<50x50x50x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][0,2][0,2]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][0,2][0,2]>, replace_indices = #modeling<index_set {[0,2][0,2][0,2]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[z][i][j] = z[i][j][z]
    %t1 = modelica.equation_template inductions = [%i0, %i1, %i2] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<50x50x50x!modelica.real>
        %1 = modelica.variable_get @z : !modelica.array<50x50x50x!modelica.real>
        %2 = modelica.load %0[%i2, %i0, %i1] : !modelica.array<50x50x50x!modelica.real>
        %3 = modelica.load %1[%i0, %i1, %i2] : !modelica.array<50x50x50x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,2][0,2][0,2]>} : !modelica.equation
}
