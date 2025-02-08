// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters --canonicalize | FileCheck %s

// CHECK-LABEL: @promotableSCC

bmodelica.model @promotableSCC {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, parameter>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.variable_get @x : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK:     bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]]
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]]

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// CHECK-LABEL: @timeDependency

bmodelica.model @timeDependency {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // COM: x = time
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.time : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK-NOT: bmodelica.initial

    // CHECK: bmodelica.dynamic
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]]

    // CHECK-NOT: bmodelica.matched_equation_instance
}
