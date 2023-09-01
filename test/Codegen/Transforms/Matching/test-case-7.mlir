// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// x[0] = 0
// x[1] + y = 0
// y + z = 0
// y + z = 0

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
    modelica.variable @z : !modelica.variable<!modelica.real>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t0 : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.constant 1 : index
        %3 = modelica.load %0[%2] : !modelica.array<2x!modelica.real>
        %4 = modelica.add %3, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0, 0]>}
    modelica.equation_instance %t1 : !modelica.equation

    // CHECK: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.real
        %1 = modelica.variable_get @z : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // CHECK-DAG: modelica.matched_equation_instance %[[t2]] {path = #modelica<equation_path [L, 0, 0]>}
    // CHECK-DAG: modelica.matched_equation_instance %[[t2]] {path = #modelica<equation_path [L, 0, 1]>}
    modelica.equation_instance %t2 : !modelica.equation
    modelica.equation_instance %t2 : !modelica.equation
}
