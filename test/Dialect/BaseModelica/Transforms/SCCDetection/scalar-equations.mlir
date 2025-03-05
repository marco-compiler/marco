// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // COM: x = y
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = 1 - x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 1>
        %2 = bmodelica.variable_get @x : !bmodelica.int
        %3 = bmodelica.sub %1, %2 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, indices = {}, match = @x
        bmodelica.matched_equation_instance %t1, indices = {}, match = @y
    }

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], indices = {}, match = @x
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @y

    // CHECK-NOT: bmodelica.matched_equation_instance
}
