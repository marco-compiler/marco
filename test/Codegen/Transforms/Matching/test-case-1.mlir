// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:1
//   x[i] + x[i + 1]
// x[2] = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.real>

    // x[i] + x[i + 1]
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.real>
        %2 = modelica.constant 1 : index
        %3 = modelica.add %i0, %2 : (index, index) -> index
        %4 = modelica.load %0[%3] : !modelica.array<3x!modelica.real>
        %5 = modelica.add %1, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<0.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[2] = 0
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.real>
        %1 = modelica.constant 2 : index
        %2 = modelica.load %0[%1] : !modelica.array<3x!modelica.real>
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,1]>, path = #modelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
        modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,1]>} : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}
