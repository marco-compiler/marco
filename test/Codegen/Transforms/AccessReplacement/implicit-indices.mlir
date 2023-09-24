// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

// Source not subscribed.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    // CHECK-DAG:       %[[array:.*]] = modelica.array_broadcast %[[zero]] : !modelica.real -> !modelica.array<3x!modelica.real>
    // CHECK-DAG:       %[[load_array:.*]] = modelica.load %[[array]][%[[i0]]]
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[x]][%[[i0]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[load_array]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i] = y[i]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<3x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<3x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3: tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,2]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,2]>, replace_indices = #modeling<index_set {[0,2]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y = {0, 0, 0}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %1 = modelica.constant #modelica.real<0.0>
        %2 = modelica.array_broadcast %1 : !modelica.real -> !modelica.array<3x!modelica.real>
        %3 = modelica.equation_side %0 : tuple<!modelica.array<3x!modelica.real>>
        %4 = modelica.equation_side %2: tuple<!modelica.array<3x!modelica.real>>
        modelica.equation_sides %3, %4 : tuple<!modelica.array<3x!modelica.real>>, tuple<!modelica.array<3x!modelica.real>>
    }

    modelica.equation_instance %t1 {id = "eq1", implicit_indices = #modeling<multidim_range [0,2]>} : !modelica.equation
}

// -----

// Source partially subscribed.

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x3x!modelica.real>
    modelica.variable @y : !modelica.variable<2x3x!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
    // CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    // CHECK-DAG:       %[[array:.*]] = modelica.array_broadcast %[[zero]] : !modelica.real -> !modelica.array<3x!modelica.real>
    // CHECK-DAG:       %[[load_array:.*]] = modelica.load %[[array]][%[[i1]]]
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[load_array]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x[i][j] = y[i][j]
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x3x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<2x3x!modelica.real>
        %2 = modelica.load %0[%i0, %i1] : !modelica.array<2x3x!modelica.real>
        %3 = modelica.load %1[%i0, %i1] : !modelica.array<2x3x!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3: tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.equation_instance %[[t0]] {id = "eq0", indices = #modeling<multidim_range [0,1][0,2]>}
    modelica.equation_instance %t0 {id = "eq0", indices = #modeling<multidim_range [0,1][0,2]>, replace_indices = #modeling<index_set {[0,1][0,2]}>, replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

    // y[i] = {0, 0, 0}
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<2x3x!modelica.real>
        %1 = modelica.subscription %0[%i0] : !modelica.array<2x3x!modelica.real>, index -> !modelica.array<3x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.array_broadcast %2 : !modelica.real -> !modelica.array<3x!modelica.real>
        %4 = modelica.equation_side %1 : tuple<!modelica.array<3x!modelica.real>>
        %5 = modelica.equation_side %3: tuple<!modelica.array<3x!modelica.real>>
        modelica.equation_sides %4, %5 : tuple<!modelica.array<3x!modelica.real>>, tuple<!modelica.array<3x!modelica.real>>
    }

    modelica.equation_instance %t1 {id = "eq1", indices = #modeling<multidim_range [0,1]>, implicit_indices = #modeling<multidim_range [0,2]>} : !modelica.equation
}
