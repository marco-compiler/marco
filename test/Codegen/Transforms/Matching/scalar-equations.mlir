// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// y = x
// y = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    // y = x
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y = 0
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [R, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}

// -----

// x[0] = x[1]
// x[0] = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.int>

    // x[0] = x[1]
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %4 = modelica.load %0[%2] : !modelica.array<2x!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        %6 = modelica.equation_side %4 : tuple<!modelica.int>
        modelica.equation_sides %5, %6 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // x[0] = 0
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<0>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [R, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}
