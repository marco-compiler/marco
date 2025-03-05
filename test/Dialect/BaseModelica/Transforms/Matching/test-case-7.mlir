// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: x[0] = 0
// COM: x[1] + y = 0
// COM: y + z = 0
// COM: y + z = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>

    // COM: x[0] = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: x[1] + y = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.tensor_extract %0[%2] : tensor<2x!bmodelica.real>
        %4 = bmodelica.add %3, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica<real 0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: y + z = 0
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @z : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {}
        bmodelica.equation_instance %t1, indices = {}
        bmodelica.equation_instance %t2, indices = {}
        bmodelica.equation_instance %t2, indices = {}

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,0]}>
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@x, {[1,1]}>
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]], match = @y
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]], match = @z
    }
}
