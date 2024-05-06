// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// i = 1 to 2
//   x[i - 1] = y[i - 1]
// x[1] = 3
// y[0] = 1

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<2x!bmodelica.int>

    // x[i - 1] = y[i - 1]
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.constant 1 : index
        %1 = bmodelica.sub %i0, %0 : (index, index) -> index
        %2 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.int>
        %3 = bmodelica.load %2[%1] : !bmodelica.array<2x!bmodelica.int>
        %4 = bmodelica.variable_get @y : !bmodelica.array<2x!bmodelica.int>
        %5 = bmodelica.load %4[%1] : !bmodelica.array<2x!bmodelica.int>
        %6 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x[1] = 3
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.int>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica.int<3>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[0] = 1
    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<2x!bmodelica.int>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<2x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica.int<1>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [1,1]>, path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2]>, path = #bmodelica<equation_path [R, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [1,2]>} : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
        bmodelica.equation_instance %t2 : !bmodelica.equation
    }
}
