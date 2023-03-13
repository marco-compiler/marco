// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(schedule{model-name=Test process-ic-model=false debug-view=true})" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}], schedule = [{block = 0 : i64, cycle = false, direction = "forward", indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}], schedule = [{block = 1 : i64, cycle = false, direction = "forward", indices = [[[0, 0]]], path = ["L"]}]}

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
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
