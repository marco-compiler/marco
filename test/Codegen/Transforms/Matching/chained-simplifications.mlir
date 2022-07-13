// RUN: modelica-opt %s --split-input-file --matching | FileCheck %s

// CHECK-DAG{LITERAL}: {id = 0 : i64, matched_indices = [[0, 0]], matched_path = ["L"]}
// CHECK-DAG{LITERAL}: {id = 1 : i64, matched_indices = [[0, 0]], matched_path = ["R"]}
// CHECK-DAG{LITERAL}: {id = 2 : i64, matched_indices = [[0, 0]], matched_path = ["R"]}

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    %2 = modelica.member_create @z : !modelica.member<!modelica.int>
    modelica.yield %0, %1, %2 : !modelica.member<!modelica.int>, !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>, %arg2: !modelica.array<!modelica.int>):
    modelica.equation attributes {id = 0} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2} {
        %0 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg2[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
