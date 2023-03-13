// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(match{model-name=Test process-ic-model=false debug-view=true})" | FileCheck %s

// y = x;
// y = 0;

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["R"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    modelica.equation attributes {id = 0} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}

// -----

// x[0] = x[1];
// x[0] = 0;

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["R"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.int>

    modelica.equation attributes {id = 0} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %4 = modelica.load %0[%2] : !modelica.array<2x!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        %6 = modelica.equation_side %4 : tuple<!modelica.int>
        modelica.equation_sides %5, %6 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.int>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.int>
        %3 = modelica.constant #modelica.int<0>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
