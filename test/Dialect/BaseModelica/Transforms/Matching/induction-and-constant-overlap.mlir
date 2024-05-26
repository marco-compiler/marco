// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// i = 1 to 5
//   x[i] = 3 - x[2]

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    // x[i] = 3 - x[2]
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 3.0>
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<5x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>} : !bmodelica.equation
    }
}
