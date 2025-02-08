// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: for i in 0:1
// COM:   x[i] = 0
// COM: for i in 2:3
// COM:   x[i] = 0

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<4x!bmodelica.real>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<4x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[i] = 0
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<4x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,1]>}
        bmodelica.equation_instance %t1 {indices = #modeling<multidim_range [2,3]>}

        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1]>, path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [2,3]>, path = #bmodelica<equation_path [L, 0]>}
    }
}
