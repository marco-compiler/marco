// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:1
//   x[i] + x[i + 1]
// x[2] = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>

    // x[i] + x[i + 1] = 0
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.add %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<3x!bmodelica.real>
        %5 = bmodelica.add %1, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.constant #bmodelica<real 0.0>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[2] = 0
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.real>
        %1 = bmodelica.constant 2 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<3x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1]>, path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,1]>}
        bmodelica.equation_instance %t1
    }
}
