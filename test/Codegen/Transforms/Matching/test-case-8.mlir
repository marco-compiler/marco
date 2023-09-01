// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:2
//   x[i] + y[0] = 0
// for i in 3:6
//   x[i] + y[1] = 0
// for i in 7:8
//   x[i] + y[2] = 0
// for i in 0:2
//   y[i] = 12

modelica.model @Test {
    modelica.variable @x : !modelica.variable<9x!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<9x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<9x!modelica.real>
        %3 = modelica.constant 0 : index
        %4 = modelica.load %1[%3] : !modelica.array<3x!modelica.real>
        %5 = modelica.add %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<10.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>} : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<9x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<9x!modelica.real>
        %3 = modelica.constant 1 : index
        %4 = modelica.load %1[%3] : !modelica.array<3x!modelica.real>
        %5 = modelica.add %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<10.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [3,6]>, path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t1 {indices = #modeling<multidim_range [3,6]>} : !modelica.equation

    // CHECK: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @x : !modelica.array<9x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<9x!modelica.real>
        %3 = modelica.constant 2 : index
        %4 = modelica.load %1[%3] : !modelica.array<3x!modelica.real>
        %5 = modelica.add %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<10.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [7,8]>, path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t2 {indices = #modeling<multidim_range [7,8]>} : !modelica.equation

    // CHECK: %[[t3:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}
    %t3 = modelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.real>
        %2 = modelica.constant #modelica.real<12.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t3]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t3 {indices = #modeling<multidim_range [0,2]>} : !modelica.equation
}
