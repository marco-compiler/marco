// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// Array dependency.

// CHECK-LABEL: @Test
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

// CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    // x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i] : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y = x[0]
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant 0 : index
        %3 = bmodelica.tensor_extract %1[%2] : tensor<5x!bmodelica.real>
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
