// RUN: modelica-opt %s --split-input-file --split-overlapping-accesses --canonicalize | FileCheck %s

// CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}, %{{.*}}] attributes {id = "t0"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1][0,7]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][0,1]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][2,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2][3,7]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [3,6][0,7]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<7x8x!modelica.real>

    // x[i0,i1] = 2 * x[2,2] - 4
    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<7x8x!modelica.real>
        %1 = modelica.load %0[%i0, %i1] : !modelica.array<7x8x!modelica.real>
        %2 = modelica.constant 2 : index
        %3 = modelica.load %0[%2, %2] : !modelica.array<7x8x!modelica.real>
        %4 = modelica.constant #modelica.int<2>
        %5 = modelica.constant #modelica.int<4>
        %6 = modelica.mul %4, %3 : (!modelica.int, !modelica.real) -> !modelica.real
        %7 = modelica.sub %6, %5 : (!modelica.real, !modelica.int) -> !modelica.real
        %8 = modelica.equation_side %1 : tuple<!modelica.real>
        %9 = modelica.equation_side %7 : tuple<!modelica.real>
        modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,6][0,7]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
