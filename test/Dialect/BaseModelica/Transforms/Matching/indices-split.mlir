// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: i = 1 to 2
// COM:   x[i - 1] = y[i - 1]
// COM: x[1] = 3
// COM: y[0] = 1

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<2x!bmodelica.int>

    // COM: x[i - 1] = y[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.constant 1 : index
        %1 = bmodelica.sub %i0, %0 : (index, index) -> index
        %2 = bmodelica.variable_get @x : tensor<2x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%1] : tensor<2x!bmodelica.int>
        %4 = bmodelica.variable_get @y : tensor<2x!bmodelica.int>
        %5 = bmodelica.tensor_extract %4[%1] : tensor<2x!bmodelica.int>
        %6 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[1] = 3
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.int>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 3>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: y[0] = 1
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : tensor<2x!bmodelica.int>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 1>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[1,2]}
        bmodelica.equation_instance %t1, indices = {}
        bmodelica.equation_instance %t2, indices = {}

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,0]}> {indices = #modeling<multidim_range [1,1]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@y, {[1,1]}> {indices = #modeling<multidim_range [2,2]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@x, {[1,1]}>
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]], match = <@y, {[0,0]}>
    }
}
