// RUN: modelica-opt %s --split-input-file --result-buffers-to-args | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

modelica.function @callee() -> (!modelica.array<heap, 3x!modelica.int>) attributes {args_names = [], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, 3x!modelica.int>
    modelica.function_terminator
}

// CHECK-LABEL: @caller
// CHECK: %[[BUFFER:[a-zA-Z0-9]*]] = modelica.alloca
// CHECK-SAME: !modelica.array<stack, 3x!modelica.int>
// CHECK: %[[ARG:[a-zA-Z0-9]*]] = modelica.array_cast %[[BUFFER]] : !modelica.array<3x!modelica.int>
// CHECK: modelica.call @callee(%[[ARG]])
// CHECK-SAME: moved_results = 1
// CHECK-SAME: (!modelica.array<3x!modelica.int>) -> ()

modelica.function @caller() -> () attributes {args_names = [], results_names = []} {
    %0 = modelica.call @callee() : () -> (!modelica.array<heap, 3x!modelica.int>)
    modelica.function_terminator
}
