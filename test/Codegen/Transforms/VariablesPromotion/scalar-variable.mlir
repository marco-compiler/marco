// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(promote-variables-to-parameters{model-name=Test})" | FileCheck %s

// Variable depending on a constant.

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int, constant>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on a parameter.

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @y : !modelica.member<!modelica.int, constant>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, constant>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    modelica.yield %0, %1 : !modelica.member<!modelica.int, constant>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>):
    modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on time.

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.time : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.real>
    }
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @y : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @z : !modelica.member<!modelica.int, constant>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, constant>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    %2 = modelica.member_create @z : !modelica.member<!modelica.int>
    modelica.yield %0, %1, %2 : !modelica.member<!modelica.int, constant>, !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>, %arg2: !modelica.array<!modelica.int>):
    modelica.initial_equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg2[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Promotable SCC

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @y : !modelica.member<!modelica.int, constant>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    modelica.yield %0, %1 : !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Promotable SCC depending on a promotable variable

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @y : !modelica.member<!modelica.int, constant>
// CHECK: modelica.member_create @z : !modelica.member<!modelica.int, constant>
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.initial_equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    %2 = modelica.member_create @z : !modelica.member<!modelica.int>
    modelica.yield %0, %1, %2 : !modelica.member<!modelica.int>, !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>, %arg2: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.load %arg2[] : !modelica.array<!modelica.int>
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg2[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %3 = modelica.add %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK: modelica.member_create @x : !modelica.member<!modelica.int>
// CHECK: modelica.member_create @y : !modelica.member<!modelica.int>
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    modelica.yield %0, %1 : !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
