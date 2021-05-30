// RUN: modelica-opt %s --result-buffers-to-args | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

modelica.function @callee() -> (!modelica.ptr<heap, 3x!modelica.int>) attributes {args_names = [], results_names = ["y"]} {
    %0 = modelica.member_create : !modelica.member<heap, 3x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.ptr<heap, 3x!modelica.int>
    modelica.return %1 : !modelica.ptr<heap, 3x!modelica.int>
}

// CHECK-LABEL: @caller
// CHECK: %[[BUFFER:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.ptr<stack, 3x!modelica.int>
// CHECK: %[[ARG:[a-zA-Z0-9]*]] = modelica.ptr_cast %[[BUFFER]] : !modelica.ptr<3x!modelica.int>
// CHECK: modelica.call @callee(%[[ARG]])
// CHECK-SAME: moved_results = 1
// CHECK-SAME: (!modelica.ptr<3x!modelica.int>) -> ()

modelica.function @caller() -> () attributes {args_names = [], results_names = []} {
    %0 = modelica.call @callee() : () -> (!modelica.ptr<heap, 3x!modelica.int>)
    modelica.return
}
