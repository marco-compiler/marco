// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x4x!modelica.real>
    modelica.variable @y : !modelica.variable<3x4x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x4x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3: tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [2,2][3,3]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [2,2][3,3]>, replace_indices = #modeling<index_set {[2,2][3,3]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[2, 3] = 0
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %1 = modelica.constant 2 : index
        %2 = modelica.constant 3 : index
        %3 = modelica.load %0[%1, %2] : !modelica.array<3x4x!modelica.real>
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4: tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t1 {id = "eq1"} : !modelica.equation
}

// -----

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x4x!modelica.real>
    modelica.variable @y : !modelica.variable<3x4x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[i0]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x4x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3: tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2][3,3]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2][3,3]>, replace_indices = #modeling<index_set {[0,2][3,3]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[i, 3] = i
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.load %0[%i0, %1] : !modelica.array<3x4x!modelica.real>
        %3 = modelica.equation_side %2 : tuple<!modelica.real>
        %4 = modelica.equation_side %i0: tuple<index>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<index>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,3]>} : !modelica.equation
}

// -----

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x4x!modelica.real>
    modelica.variable @y : !modelica.variable<3x4x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[i1]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i, j] = y[i, j]
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x4x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<3x4x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3: tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [2,2][0,3]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [2,2][0,3]>, replace_indices = #modeling<index_set {[2,2][0,3]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[2, i] = i
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x4x!modelica.real>
        %1 = modelica.constant 2 : index
        %2 = modelica.load %0[%1, %i0] : !modelica.array<3x4x!modelica.real>
        %3 = modelica.equation_side %2 : tuple<!modelica.real>
        %4 = modelica.equation_side %i0: tuple<index>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<index>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,3]>} : !modelica.equation
}
