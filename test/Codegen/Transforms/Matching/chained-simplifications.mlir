// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// x = 0
// x = y
// y = z

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int>

    // x = 0
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x = y
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y = z
    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.variable_get @z : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.main_model {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [R, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {path = #bmodelica<equation_path [R, 0]>}
        bmodelica.equation_instance %t0 : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
        bmodelica.equation_instance %t2 : !bmodelica.equation
    }
}
