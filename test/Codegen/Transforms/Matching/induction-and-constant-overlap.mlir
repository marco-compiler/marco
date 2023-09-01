// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// i = 1 to 5
//   x[i] = 3 - x[2];

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %2 = modelica.constant #modelica.real<3.0>
        %3 = modelica.constant 2 : index
        %4 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %5 = modelica.sub %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.equation_side %1 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
}
