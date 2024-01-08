// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:1
//   x[i] - y[i] = 0
// x[0] + x[1] = 2
// y[0] + y[1] = 3

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>
    modelica.variable @y : !modelica.variable<2x!modelica.real>

    // x[i] - y[i] = 0
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<2x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<2x!modelica.real>
        %4 = modelica.sub %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[0] + x[1] = 2
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %4 = modelica.load %0[%2] : !modelica.array<2x!modelica.real>
        %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<2.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y[0] + y[1] = 3
    // CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %4 = modelica.load %0[%2] : !modelica.array<2x!modelica.real>
        %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.constant #modelica.real<3.0>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t2]]
        modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,1]>} : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
        modelica.equation_instance %t2 : !modelica.equation
    }
}
