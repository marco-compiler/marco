// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: for i in 0:2
// COM:   x[i] + y[0] = 0
// COM: for i in 3:6
// COM:   x[i] + y[1] = 0
// COM: for i in 7:8
// COM:   x[i] + y[2] = 0
// COM: for i in 0:2
// COM:   y[i] = 12

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<9x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>

    // COM: x[i] + y[0] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<9x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<9x!bmodelica.real>
        %3 = bmodelica.constant 0 : index
        %4 = bmodelica.tensor_extract %1[%3] : tensor<3x!bmodelica.real>
        %5 = bmodelica.add %2, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 10.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[i] + y[1] = 0
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<9x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<9x!bmodelica.real>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.tensor_extract %1[%3] : tensor<3x!bmodelica.real>
        %5 = bmodelica.add %2, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 10.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    // COM: x[i] + y[2] = 0
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @x : tensor<9x!bmodelica.real>
        %1 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<9x!bmodelica.real>
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.tensor_extract %1[%3] : tensor<3x!bmodelica.real>
        %5 = bmodelica.add %2, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 10.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    // COM: y[i] = 12
    %t3 = bmodelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 12.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>}
        bmodelica.equation_instance %t1 {indices = #modeling<multidim_range [3,6]>}
        bmodelica.equation_instance %t2 {indices = #modeling<multidim_range [7,8]>}
        bmodelica.equation_instance %t3 {indices = #modeling<multidim_range [0,2]>}

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [3,6]>, path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [7,8]>, path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t3]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
    }
}
