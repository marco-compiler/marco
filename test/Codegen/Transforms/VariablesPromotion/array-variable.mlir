// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(promote-variables-to-parameters{model-name=Test})" | FileCheck %s

// Variable depending on a constant.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.constant #modelica.int<0>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Variable depending on a parameter.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<3x!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
    modelica.variable @y : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.constant #modelica.int<0>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 1, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Variable depending on time.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.time : !modelica.real
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.real>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.real>
        }
    }
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @z : !modelica.member<3x!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
    modelica.variable @y : !modelica.member<3x!modelica.int>
    modelica.variable @z : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.constant #modelica.int<0>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 1, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 2, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Promotable SCC.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<3x!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>
    modelica.variable @y : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 1, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Promotable SCC depending on a promotable variable.

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @y : !modelica.member<3x!modelica.int, parameter>
// CHECK: modelica.variable @z : !modelica.member<3x!modelica.int, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>
    modelica.variable @y : !modelica.member<3x!modelica.int>
    modelica.variable @z : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.constant #modelica.int<0>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 1, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
            %5 = modelica.load %4[%i] : !modelica.array<3x!modelica.int>
            %6 = modelica.add %3, %5 : (!modelica.int, !modelica.int) -> !modelica.int
            %7 = modelica.equation_side %1 : tuple<!modelica.int>
            %8 = modelica.equation_side %6 : tuple<!modelica.int>
            modelica.equation_sides %7, %8 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 2, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %5 = modelica.load %4[%i] : !modelica.array<3x!modelica.int>
            %6 = modelica.add %3, %5 : (!modelica.int, !modelica.int) -> !modelica.int
            %7 = modelica.equation_side %1 : tuple<!modelica.int>
            %8 = modelica.equation_side %6 : tuple<!modelica.int>
            modelica.equation_sides %7, %8 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK: modelica.variable @x : !modelica.member<3x!modelica.int>
// CHECK: modelica.variable @y : !modelica.member<3x!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 2]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>
    modelica.variable @y : !modelica.member<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 2]]], path = ["L"]}]} {
            %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Promotable array written by different equations.

// CHECK: modelica.variable @x : !modelica.member<2x!modelica.real, parameter>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[1, 1]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<2x!modelica.real>

    modelica.for_equation %i = 0 to 0 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
            %1 = modelica.load %0[%i] : !modelica.array<2x!modelica.real>
            %2 = modelica.constant #modelica.real<0.0>
            %3 = modelica.equation_side %1 : tuple<!modelica.real>
            %4 = modelica.equation_side %2 : tuple<!modelica.real>
            modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }

    modelica.for_equation %i = 1 to 1 {
        modelica.equation attributes {id = 1, match = [{indices = [[[1, 1]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
            %1 = modelica.load %0[%i] : !modelica.array<2x!modelica.real>
            %2 = modelica.constant #modelica.real<1.0>
            %3 = modelica.equation_side %1 : tuple<!modelica.real>
            %4 = modelica.equation_side %2 : tuple<!modelica.real>
            modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}


// -----

// Array not fully promotable.

// CHECK: modelica.variable @x : !modelica.member<2x!modelica.real>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[1, 1]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<2x!modelica.real>

    modelica.for_equation %i = 0 to 0 {
        modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
            %1 = modelica.load %0[%i] : !modelica.array<2x!modelica.real>
            %2 = modelica.constant #modelica.real<0.0>
            %3 = modelica.equation_side %1 : tuple<!modelica.real>
            %4 = modelica.equation_side %2 : tuple<!modelica.real>
            modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }

    modelica.for_equation %i = 1 to 1 {
        modelica.equation attributes {id = 1, match = [{indices = [[[1, 1]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
            %1 = modelica.load %0[%i] : !modelica.array<2x!modelica.real>
            %2 = modelica.time : !modelica.real
            %3 = modelica.equation_side %1 : tuple<!modelica.real>
            %4 = modelica.equation_side %2 : tuple<!modelica.real>
            modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}
