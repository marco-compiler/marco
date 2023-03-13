// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(schedule{model-name=Test process-ic-model=false debug-view=true})" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 1]]], path = ["L"]}], schedule = [{block = 1 : i64, cycle = false, direction = "backward", indices = [[[0, 1]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}], schedule = [{block = 0 : i64, cycle = false, direction = "forward", indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.for_equation %arg1 = 0 to 1 {
        modelica.equation attributes {id = 0, match = [{indices=[[[0, 1]]], path = ["L"]}]} {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%arg1] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %3 = modelica.constant 1 : index
            %4 = modelica.add %arg1, %3 : (index, index) -> index
            %5 = modelica.load %2[%4] : !modelica.array<3x!modelica.int>
            %6 = modelica.equation_side %1 : tuple<!modelica.int>
            %7 = modelica.equation_side %5 : tuple<!modelica.int>
            modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.constant 2 : index
        %2 = modelica.load %0[%1] : !modelica.array<3x!modelica.int>
        %3 = modelica.constant #modelica.int<0>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
