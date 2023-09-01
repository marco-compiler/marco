// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// l + f[0] = 0
// f[0] = 0
// for i in 0:4
//   x[i] + f[i] + f[i + 1] = 0
// for i in 1:4
//   f[i] = 0
// h + f[5] = 0
// f[5] = 0

modelica.model @Test {
    modelica.variable @l : !modelica.variable<!modelica.real>
    modelica.variable @h : !modelica.variable<!modelica.real>
    modelica.variable @x : !modelica.variable<5x!modelica.real>
    modelica.variable @f : !modelica.variable<6x!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @l : !modelica.real
        %1 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %2 = modelica.constant 0 : index
        %3 = modelica.load %1[%2] : !modelica.array<6x!modelica.real>
        %4 = modelica.add %0, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t0 : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<6x!modelica.real>
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t1 : !modelica.equation

    // CHECK: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<6x!modelica.real>
        %4 = modelica.constant 1 : index
        %5 = modelica.add %i0, %4 : (index, index) -> index
        %6 = modelica.load %1[%5] : !modelica.array<6x!modelica.real>
        %7 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %8 = modelica.add %7, %6 : (!modelica.real, !modelica.real) -> !modelica.real
        %9 = modelica.constant #modelica.real<0.0>
        %10 = modelica.equation_side %8 : tuple<!modelica.real>
        %11 = modelica.equation_side %9 : tuple<!modelica.real>
        modelica.equation_sides %10, %11 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0, 0, 0]>}
    modelica.equation_instance %t2 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation

    // CHECK: %[[t3:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}
    %t3 = modelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %0 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<6x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t3]] {indices = #modeling<multidim_range [1,4]>, path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t3 {indices = #modeling<multidim_range [1,4]>} : !modelica.equation

    // CHECK: %[[t4:.*]] = modelica.equation_template inductions = [] attributes {id = "t4"}
    %t4 = modelica.equation_template inductions = [] attributes {id = "t4"} {
        %0 = modelica.variable_get @h : !modelica.real
        %1 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %2 = modelica.constant 5 : index
        %3 = modelica.load %1[%2] : !modelica.array<6x!modelica.real>
        %4 = modelica.add %0, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t4]] {path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t4 : !modelica.equation

    // CHECK: %[[t5:.*]] = modelica.equation_template inductions = [] attributes {id = "t5"}
    %t5 = modelica.equation_template inductions = [] attributes {id = "t5"} {
        %0 = modelica.variable_get @f : !modelica.array<6x!modelica.real>
        %1 = modelica.constant 5 : index
        %2 = modelica.load %0[%1] : !modelica.array<6x!modelica.real>
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t5]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t5 : !modelica.equation
}
