// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters --canonicalize | FileCheck %s

// Promotable SCC.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK: bmodelica.initial
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NOT: bmodelica.dynamic

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.variable_get @x : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
