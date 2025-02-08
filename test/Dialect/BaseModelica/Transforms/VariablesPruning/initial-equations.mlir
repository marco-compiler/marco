// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-DAG:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // COM: x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: z = y
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t2 {path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK:       bmodelica.initial {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t0]] {{.*$}}
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]] {{.*$}}
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t2]] {{.*$}}
    // CHECK-NEXT:  }
}
