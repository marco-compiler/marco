// RUN: modelica-opt %s --split-input-file --test-access-replacement --canonicalize | FileCheck %s

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>

    // CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"} {
    // CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
    // CHECK-DAG:       %[[value:.*]] = modelica.constant  #modelica.real<0.000000e+00>
    // CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
    // CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[value]]
    // CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // x = y
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1: tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y = 0
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.real
        %1 = modelica.constant #modelica.real<0.0>
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1: tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK: modelica.equation_instance %[[t0]] {id = "eq0"}
        modelica.equation_instance %t0 {id = "eq0", replace_destination_path = #modelica<equation_path [R, 0]>, replace_eq = "eq1", replace_source_path = #modelica<equation_path [L, 0]>} : !modelica.equation

        modelica.equation_instance %t1 {id = "eq1"} : !modelica.equation
    }
}
