// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[value:.*]] = bmodelica.constant  #bmodelica<real 0.000000e+00>
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[value]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x = y
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1: tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1: tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK: bmodelica.equation_instance %[[t0]], indices = {} {id = "eq0"}
        bmodelica.equation_instance %t0, indices = {} {id = "eq0", replace_destination_path = #bmodelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #bmodelica<equation_path [L, 0]>}

        bmodelica.equation_instance %t1, indices = {} {id = "eq1"}
    }
}
