// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(promote-variables-to-parameters{model-name=Test})" | FileCheck %s

// Variable depending on a constant.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on a parameter.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int, parameter>
    modelica.variable @y : !modelica.member<!modelica.int>

    modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on time.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.time : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.real>
    }
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @z : !modelica.member<!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int, parameter>
    modelica.variable @y : !modelica.member<!modelica.int>
    modelica.variable @z : !modelica.member<!modelica.int>

    modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Promotable SCC.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>
    modelica.variable @y : !modelica.member<!modelica.int>

    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Promotable SCC depending on a promotable variable.

// CHECK: modelica.variable @x : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int, parameter>
// CHECK: modelica.variable @z : !modelica.member<!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>
    modelica.variable @y : !modelica.member<!modelica.int>
    modelica.variable @z : !modelica.member<!modelica.int>

    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.variable_get @z : !modelica.int
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.variable_get @y : !modelica.int
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK: modelica.variable @x : !modelica.member<!modelica.int>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>
    modelica.variable @y : !modelica.member<!modelica.int>

    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
