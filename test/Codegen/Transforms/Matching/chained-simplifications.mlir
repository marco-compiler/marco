// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// x = 0;
// x = y;
// y = z;

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>
    modelica.variable @z : !modelica.variable<!modelica.int>

    // CHECK: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}
    modelica.equation_instance %t0 : !modelica.equation

    // CHECK: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [R, 0]>}
    modelica.equation_instance %t1 : !modelica.equation

    // CHECK: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @z : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // CHECK: modelica.matched_equation_instance %[[t2]] {path = #modelica<equation_path [R, 0]>}
    modelica.equation_instance %t2 : !modelica.equation
}
