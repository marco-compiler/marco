// RUN: modelica-opt %s --split-input-file --split-overlapping-accesses --canonicalize | FileCheck %s

// CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}, %{{.*}}] attributes {id = "t0"}
// CHECK: bmodelica.dynamic
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1][0,7]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][0,1]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][2,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][3,7]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [3,6][0,7]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<7x8x!bmodelica.real>

    // x[i0,i1] = 2 * x[2,2] - 4
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<7x8x!bmodelica.real>
        %1 = bmodelica.load %0[%i0, %i1] : !bmodelica.array<7x8x!bmodelica.real>
        %2 = bmodelica.constant 2 : index
        %3 = bmodelica.load %0[%2, %2] : !bmodelica.array<7x8x!bmodelica.real>
        %4 = bmodelica.constant #bmodelica.int<2>
        %5 = bmodelica.constant #bmodelica.int<4>
        %6 = bmodelica.mul %4, %3 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.sub %6, %5 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
        %8 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,6][0,7]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
