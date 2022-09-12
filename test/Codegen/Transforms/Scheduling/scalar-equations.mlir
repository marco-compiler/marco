// RUN: modelica-opt %s --split-input-file --pass-pipeline="schedule{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}], schedule = [{block = 0 : i64, cycle = false, direction = "forward", indices = [[[0, 0]]]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}], schedule = [{block = 1 : i64, cycle = false, direction = "forward", indices = [[[0, 0]]]}]}

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
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
