// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    // COM: x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // COM: y = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = @x
        bmodelica.matched_equation_instance %t1, match = @y
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], match = @y
    // CHECK-NEXT:  }
}
