// RUN: modelica-opt %s --split-input-file --promote-output-arrays | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

// CHECK-LABEL: @f1
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]] : !modelica.array<3x!modelica.int>
// CHECK-SAME: -> ()
// CHECK-SAME: args_names = ["y"]
// CHECK-SAME: results_names = []
// CHECK:   modelica.store %[[ARG0]][%{{[a-zA-Z0-9]*}}], %{{[a-zA-Z0-9]*}} : !modelica.array<3x!modelica.int>

modelica.function @f1() -> (!modelica.array<heap, 3x!modelica.int>) attributes {args_names = [], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, 3x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.array<heap, 3x!modelica.int>
    %2 = modelica.constant #modelica.int<1>
    %3 = modelica.constant 0 : index
    modelica.store %1[%3], %2 : !modelica.array<heap, 3x!modelica.int>
    modelica.function_terminator
}
