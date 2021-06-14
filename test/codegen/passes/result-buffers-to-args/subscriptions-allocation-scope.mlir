// RUN: modelica-opt %s --split-input-file --result-buffers-to-args | FileCheck %s

// Subscriptions should change the allocation scope

// CHECK-LABEL: @f3
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]] : !modelica.ptr<3x!modelica.int>
// CHECK: %[[INDEX:[a-zA-Z0-9]*]] = modelica.constant 0 : index
// CHECK: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARG0]][%[[INDEX]]] : !modelica.ptr<3x!modelica.int>, index
// CHECK: modelica.store %[[SUBSCRIPTION]][], %{{[a-zA-Z0-9]*}} : !modelica.ptr<!modelica.int>

modelica.function @f3() -> (!modelica.ptr<heap, 3x!modelica.int>) attributes {args_names = [], results_names = ["y"]} {
    %0 = modelica.member_create : !modelica.member<heap, 3x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.ptr<heap, 3x!modelica.int>
    %2 = modelica.constant #modelica.int<1>
    %3 = modelica.constant 0 : index
    %4 = modelica.subscription %1[%3] : !modelica.ptr<heap, 3x!modelica.int>, index
    modelica.store %4[], %2 : !modelica.ptr<heap, !modelica.int>
    %5 = modelica.member_load %0 : !modelica.ptr<heap, 3x!modelica.int>
    modelica.return %5 : !modelica.ptr<heap, 3x!modelica.int>
}
