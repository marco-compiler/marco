// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// i = 1 to 2
// x[i - 1] = y[i - 1];

// x[1] = 3;
// y[0] = 1;

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.int>
    modelica.variable @y : !modelica.variable<2x!modelica.int>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.constant 1 : index
        %1 = modelica.sub %i0, %0 : (index, index) -> index
        %2 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %3 = modelica.load %2[%1] : !modelica.array<2x!modelica.int>
        %4 = modelica.variable_get @y : !modelica.array<2x!modelica.int>
        %5 = modelica.load %4[%1] : !modelica.array<2x!modelica.int>
        %6 = modelica.equation_side %3 : tuple<!modelica.int>
        %7 = modelica.equation_side %5 : tuple<!modelica.int>
        modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>}
    // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [2,2]>, path = #modelica<equation_path [R, 0]>}
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [1,2]>} : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 1 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<3>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t1 : !modelica.equation

    // CHECK: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<1>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK: modelica.matched_equation_instance %[[t2]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t2 : !modelica.equation
}
