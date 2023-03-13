// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(match{model-name=Test process-ic-model=false debug-view=true})" | FileCheck %s

// i = 1 to 2
// x[i - 1] = y[i - 1];

// x[1] = 3;
// y[0] = 1;

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[1, 1]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[2, 2]]], path = ["R"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.member<2x!modelica.int>
    modelica.variable @y : !modelica.member<2x!modelica.int>

    modelica.for_equation %arg2 = 1 to 2 {
        modelica.equation attributes {id = 0} {
            %0 = modelica.constant 1 : index
            %1 = modelica.sub %arg2, %0 : (index, index) -> index
            %2 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
            %3 = modelica.load %2[%1] : !modelica.array<2x!modelica.int>
            %4 = modelica.variable_get @y : !modelica.array<2x!modelica.int>
            %5 = modelica.load %4[%1] : !modelica.array<2x!modelica.int>
            %6 = modelica.equation_side %3 : tuple<!modelica.int>
            %7 = modelica.equation_side %5 : tuple<!modelica.int>
            modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.constant 1 : index
        %1 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %2 = modelica.load %1[%0] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<3>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2} {
        %0 = modelica.constant 0 : index
        %1 = modelica.variable_get @y : !modelica.array<2x!modelica.int>
        %2 = modelica.load %1[%0] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<1>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
