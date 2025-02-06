// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:4
//   x[i] = 10
// for i in 0:3
//   y[i] = x[i + 1]
// for i in 0:3
//   z[i] = x[i] + y[i]
// z[4] = x[4]

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<4x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<5x!bmodelica.real>

    // [i] = 10
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 10.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[i] = x[i + 1]
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<4x!bmodelica.real>
        %1 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<4x!bmodelica.real>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.add %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %1[%4] : tensor<5x!bmodelica.real>
        %6 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z[i] = x[i] + y[i]
    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : tensor<5x!bmodelica.real>
        %1 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %2 = bmodelica.variable_get @y : tensor<4x!bmodelica.real>
        %3 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.real>
        %4 = bmodelica.tensor_extract %1[%i0] : tensor<5x!bmodelica.real>
        %5 = bmodelica.tensor_extract %2[%i0] : tensor<4x!bmodelica.real>
        %6 = bmodelica.add %4, %5 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z[4] = x[4]
    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @z : tensor<5x!bmodelica.real>
        %1 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant 4 : index
        %3 = bmodelica.tensor_extract %0[%2] : tensor<5x!bmodelica.real>
        %4 = bmodelica.tensor_extract %1[%2] : tensor<5x!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,3]>, path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,3]>, path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t3]] {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>}
        bmodelica.equation_instance %t1 {indices = #modeling<multidim_range [0,3]>}
        bmodelica.equation_instance %t2 {indices = #modeling<multidim_range [0,3]>}
        bmodelica.equation_instance %t3
    }
}
