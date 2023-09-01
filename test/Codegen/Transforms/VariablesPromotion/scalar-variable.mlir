// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters | FileCheck %s

// Variable depending on a constant.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Variable depending on a parameter.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int, parameter>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {initial = true, path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Variable depending on time.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.time : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.real>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @z : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t2]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int, parameter>
    modelica.variable @y : !modelica.variable<!modelica.int>
    modelica.variable @z : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {initial = true, path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t2 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Promotable SCC.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Promotable SCC depending on a promotable variable.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @z : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t2]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>
    modelica.variable @z : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.variable_get @z : !modelica.int
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.variable_get @y : !modelica.int
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t2 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
