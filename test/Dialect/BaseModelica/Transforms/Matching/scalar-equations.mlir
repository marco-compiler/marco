// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// y = x
// y = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // y = x
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.variable_get @x : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y = 0
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [R, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
    }
}

// -----

// x[0] = x[1]
// x[0] = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.int>

    // x[0] = x[1]
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.int>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.int>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<2x!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.int>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x[0] = 0
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.int>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [R, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
    }
}
