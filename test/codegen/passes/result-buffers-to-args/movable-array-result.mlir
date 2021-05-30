// RUN: modelica-opt %s --result-buffers-to-args | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

// CHECK-LABEL: @f1
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]] : !modelica.ptr<3x!modelica.int>
// CHECK-SAME: -> ()
// CHECK-SAME: args_names = ["y"]
// CHECK-SAME: results_names = []
// CHECK:   modelica.store %{{[a-zA-Z0-9]*}}, %[[ARG0]][%{{[a-zA-Z0-9]*}}] : !modelica.ptr<3x!modelica.int>
// CHECK:   modelica.return

modelica.function @f1() -> (!modelica.ptr<heap, 3x!modelica.int>) attributes {args_names = [], results_names = ["y"]} {
    %0 = modelica.member_create : !modelica.member<heap, 3x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.ptr<heap, 3x!modelica.int>
    %2 = modelica.constant #modelica.int<1>
    %3 = modelica.constant 0 : index
    modelica.store %2, %1[%3] : !modelica.ptr<heap, 3x!modelica.int>
    %4 = modelica.member_load %0 : !modelica.ptr<heap, 3x!modelica.int>
    modelica.return %4 : !modelica.ptr<heap, 3x!modelica.int>
}
