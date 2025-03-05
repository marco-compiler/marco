// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COOM: y = x
// COM: y = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // COM: y = x
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.variable_get @x : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // COM: y = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = @x
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = @y
    }
}

// -----

// COM: x[0] = x[1]
// COM: x[0] = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.int>

    // COM: x[0] = x[1]
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

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: x[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.int>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[1,1]}>
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@x, {[0,0]}>
    }
}
