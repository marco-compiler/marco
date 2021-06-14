// RUN: modelica-opt %s --split-input-file --result-buffers-to-args | FileCheck %s

// Dynamic output arrays can't be moved

// CHECK-LABEL: @f2
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.ptr<heap, ?x!modelica.int>)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]

modelica.function @f2(%arg0 : !modelica.int) -> (!modelica.ptr<heap, ?x!modelica.int>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create %arg0 : !modelica.int -> !modelica.member<heap, ?x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.ptr<heap, ?x!modelica.int>
    %2 = modelica.constant #modelica.int<1>
    %3 = modelica.constant 0 : index
    modelica.store %1[%3], %2 : !modelica.ptr<heap, ?x!modelica.int>
    %4 = modelica.member_load %0 : !modelica.ptr<heap, ?x!modelica.int>
    modelica.return %4 : !modelica.ptr<heap, ?x!modelica.int>
}
