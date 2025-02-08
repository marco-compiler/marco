// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: for i in 0:2
// COM:   x[i] + y[i] = 0
// COM: for i in 0:5
// COM:   x[i] + y[1] = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<6x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>

    // COM: x[i] + y[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<6x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<6x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<3x!bmodelica.real>
        %4 = bmodelica.add %2, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica<real 0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[i] + y[1] = 0
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<6x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<6x!bmodelica.real>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.tensor_extract %1[%3] : tensor<3x!bmodelica.real>
        %5 = bmodelica.add %2, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 0.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>}
        bmodelica.equation_instance %t1 {indices = #modeling<multidim_range [0,5]>}

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]]
    }
}
