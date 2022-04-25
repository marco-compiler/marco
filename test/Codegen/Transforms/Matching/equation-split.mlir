// RUN: modelica-opt %s --split-input-file --matching | FileCheck %s

// CHECK-DAG{LITERAL}: {id = 0 : i64, matched_indices = [[1, 1]], matched_path = ["L"]}
// CHECK-DAG{LITERAL}: {id = 0 : i64, matched_indices = [[2, 2]], matched_path = ["R"]}
// CHECK-DAG{LITERAL}: {id = 1 : i64, matched_indices = [[0, 0]], matched_path = ["L"]}
// CHECK-DAG{LITERAL}: {id = 2 : i64, matched_indices = [[0, 0]], matched_path = ["L"]}

modelica.model {
    %0 = modelica.member_create @x : !modelica.member<2x!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<2x!modelica.int>
    modelica.yield %0, %1 : !modelica.member<2x!modelica.int>, !modelica.member<2x!modelica.int>
} equations {
^bb0(%arg0: !modelica.array<2x!modelica.int>, %arg1: !modelica.array<2x!modelica.int>):
    modelica.for_equation %arg2 = 1 to 2 {
        modelica.equation attributes {id = 0} {
            %0 = modelica.constant 1 : index
            %1 = modelica.sub %arg2, %0 : (index, index) -> index
            %2 = modelica.load %arg0[%1] : !modelica.array<2x!modelica.int>
            %3 = modelica.load %arg1[%1] : !modelica.array<2x!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.constant 1 : index
        %1 = modelica.load %arg0[%0] : !modelica.array<2x!modelica.int>
        %2 = modelica.constant #modelica.int<3>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation attributes {id = 2} {
        %0 = modelica.constant 0 : index
        %1 = modelica.load %arg1[%0] : !modelica.array<2x!modelica.int>
        %2 = modelica.constant #modelica.int<1>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
