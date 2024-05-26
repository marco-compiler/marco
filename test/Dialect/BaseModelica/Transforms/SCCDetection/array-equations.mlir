// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.scc {
// CHECK-NEXT:          bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:          bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:          bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [3,4]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<5x!bmodelica.int>

    // x[i] = y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<5x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = 1 - x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 1>
        %3 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %4 = bmodelica.tensor_extract %3[%i0] : tensor<5x!bmodelica.int>
        %5 = bmodelica.sub %2, %4 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = 2 - x[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 2>
        %3 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %4 = bmodelica.tensor_extract %3[%i0] : tensor<5x!bmodelica.int>
        %5 = bmodelica.sub %2, %4 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [3,4]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
