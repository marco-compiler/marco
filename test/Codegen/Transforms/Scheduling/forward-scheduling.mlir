// RUN: modelica-opt %s --split-input-file --scheduling | FileCheck %s

// CHECK{LITERAL}: {id = 1 : i64, scheduled_direction = "forward", scheduled_indices = [[0, 0]]}
// CHECK{LITERAL}: {id = 0 : i64, scheduled_direction = "forward", scheduled_indices = [[1, 2]]}

modelica.model {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int>
    modelica.yield %0 : !modelica.member<3x!modelica.int>
} equations {
^bb0(%arg0: !modelica.array<3x!modelica.int>):
    modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation attributes {id = 0} {
            %0 = modelica.load %arg0[%arg1] : !modelica.array<3x!modelica.int>
            %1 = modelica.constant 1 : index
            %2 = modelica.sub %arg1, %1 : (index, index) -> index
            %3 = modelica.load %arg0[%2] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %0 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }

    modelica.equation attributes {id = 1} {
        %0 = modelica.constant 0 : index
        %1 = modelica.load %arg0[%0] : !modelica.array<3x!modelica.int>
        %2 = modelica.constant #modelica.int<0>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
