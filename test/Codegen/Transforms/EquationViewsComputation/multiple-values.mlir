// RUN: modelica-opt %s --split-input-file --compute-equation-views | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: modelica.equation_instance %[[t0]] {view_element_index = 0 : i64}
// CHECK-DAG: modelica.equation_instance %[[t0]] {view_element_index = 1 : i64}

modelica.model @Test {
    modelica.variable @x1 : !modelica.variable<!modelica.int>
    modelica.variable @x2 : !modelica.variable<!modelica.int>
    modelica.variable @x3 : !modelica.variable<!modelica.int>
    modelica.variable @x4 : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x1 : !modelica.int
        %1 = modelica.variable_get @x2 : !modelica.int
        %2 = modelica.variable_get @x3 : !modelica.int
        %3 = modelica.variable_get @x4 : !modelica.int
        %4 = modelica.equation_side %0, %1 : tuple<!modelica.int, !modelica.int>
        %5 = modelica.equation_side %2, %3 : tuple<!modelica.int, !modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int, !modelica.int>, tuple<!modelica.int, !modelica.int>
    }

    modelica.equation_instance %t0 : !modelica.equation
}

// -----

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: modelica.equation_instance %[[t0]] {view_element_index = 0 : i64}
// CHECK-DAG: modelica.equation_instance %[[t0]] {view_element_index = 1 : i64}

modelica.model @Test {
    modelica.variable @x1 : !modelica.variable<3x!modelica.int>
    modelica.variable @x2 : !modelica.variable<3x!modelica.int>
    modelica.variable @x3 : !modelica.variable<3x!modelica.int>
    modelica.variable @x4 : !modelica.variable<3x!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x1 : !modelica.array<3x!modelica.int>
        %1 = modelica.variable_get @x2 : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x3 : !modelica.array<3x!modelica.int>
        %3 = modelica.variable_get @x4 : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %0, %1 : tuple<!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>>
        %5 = modelica.equation_side %2, %3 : tuple<!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>>
        modelica.equation_sides %4, %5 : tuple<!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>>, tuple<!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>>
    }

    modelica.equation_instance %t0 : !modelica.equation
}

// -----

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, view_element_index = 0 : i64}
// CHECK-DAG: modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, view_element_index = 1 : i64}

modelica.model @Test {
    modelica.variable @x1 : !modelica.variable<3x5x!modelica.int>
    modelica.variable @x2 : !modelica.variable<3x5x!modelica.int>
    modelica.variable @x3 : !modelica.variable<3x5x!modelica.int>
    modelica.variable @x4 : !modelica.variable<3x5x!modelica.int>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x1 : !modelica.array<3x5x!modelica.int>
        %1 = modelica.variable_get @x2 : !modelica.array<3x5x!modelica.int>
        %2 = modelica.variable_get @x3 : !modelica.array<3x5x!modelica.int>
        %3 = modelica.variable_get @x4 : !modelica.array<3x5x!modelica.int>
        %4 = modelica.subscription %0[%i0] : !modelica.array<3x5x!modelica.int>, index -> !modelica.array<5x!modelica.int>
        %5 = modelica.subscription %1[%i0] : !modelica.array<3x5x!modelica.int>, index -> !modelica.array<5x!modelica.int>
        %6 = modelica.subscription %2[%i0] : !modelica.array<3x5x!modelica.int>, index -> !modelica.array<5x!modelica.int>
        %7 = modelica.subscription %3[%i0] : !modelica.array<3x5x!modelica.int>, index -> !modelica.array<5x!modelica.int>
        %8 = modelica.equation_side %4, %5 : tuple<!modelica.array<5x!modelica.int>, !modelica.array<5x!modelica.int>>
        %9 = modelica.equation_side %6, %7 : tuple<!modelica.array<5x!modelica.int>, !modelica.array<5x!modelica.int>>
        modelica.equation_sides %8, %9 : tuple<!modelica.array<5x!modelica.int>, !modelica.array<5x!modelica.int>>, tuple<!modelica.array<5x!modelica.int>, !modelica.array<5x!modelica.int>>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>} : !modelica.equation
}
