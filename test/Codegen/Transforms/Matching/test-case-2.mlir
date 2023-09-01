// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:1
//   x[i] = 0
// for i in 2:3
//   x[i] = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<4x!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<4x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<4x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1]>, path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,1]>} : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<4x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<4x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %1 {indices = #modeling<multidim_range [2,3]>, path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t1 {indices = #modeling<multidim_range [2,3]>} : !modelica.equation
}
