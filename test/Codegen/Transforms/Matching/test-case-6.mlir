// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:2
//   x[i] + y[i] = 0
// for i in 0:5
//   x[i] + y[1] = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<6x!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<6x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<6x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<3x!modelica.real>
        %4 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]]
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,2]>} : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<6x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<6x!modelica.real>
        %3 = modelica.constant 1 : index
        %4 = modelica.load %1[%3] : !modelica.array<3x!modelica.real>
        %5 = modelica.add %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<0.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]]
    modelica.equation_instance %t1 {indices = #modeling<multidim_range [0,5]>} : !modelica.equation
}
