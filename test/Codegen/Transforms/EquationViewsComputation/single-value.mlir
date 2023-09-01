// RUN: modelica-opt %s --split-input-file --compute-equation-views | FileCheck %s

// CHECK: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK: modelica.equation_instance %[[t0]] {view_element_index = 0 : i64}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation_instance %t0 : !modelica.equation
}

// -----

// CHECK: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK: modelica.equation_instance %[[t0]] {view_element_index = 0 : i64}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.array<3x!modelica.int>>
        %3 = modelica.equation_side %1 : tuple<!modelica.array<3x!modelica.int>>
        modelica.equation_sides %2, %3 : tuple<!modelica.array<3x!modelica.int>>, tuple<!modelica.array<3x!modelica.int>>
    }

    modelica.equation_instance %t0 : !modelica.equation
}

// -----

// CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, view_element_index = 0 : i64}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %2 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %3 = modelica.load %1[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>} : !modelica.equation
}
