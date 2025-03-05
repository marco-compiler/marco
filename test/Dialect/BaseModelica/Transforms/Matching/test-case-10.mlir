// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: for i in 0:1
// COM:   x[i] - y[i] = 0
// COM: x[0] + x[1] = 2
// COM: y[0] + y[1] = 3

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<2x!bmodelica.real>

    // COM: x[i] - y[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<2x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<2x!bmodelica.real>
        %4 = bmodelica.sub %2, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica<real 0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[0] + x[1] = 2
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<2x!bmodelica.real>
        %5 = bmodelica.add %3, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 2.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: y[0] + y[1] = 3
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<2x!bmodelica.real>
        %5 = bmodelica.add %3, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 3.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[0,1]}
        bmodelica.equation_instance %t1
        bmodelica.equation_instance %t2

        // CHECK-DAG: bmodelica.equation_instance %[[t0]]
        // CHECK-DAG: bmodelica.equation_instance %[[t1]]
        // CHECK-DAG: bmodelica.equation_instance %[[t2]]
    }
}
