// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// Static output arrays can be promoted

// CHECK-LABEL: @callee
// CHECK-SAME: (%[[x:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK: %[[y:.*]] = modelica.alloc %{{.*}} : !modelica.array<?x!modelica.int>
// CHECK: modelica.raw_return %[[y]]

modelica.function @callee : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int, output>
    %1 = arith.constant 2 : index
    %2 = modelica.member_create @y %1 : !modelica.member<?x!modelica.int, output>
}

// CHECK-LABEL: @caller
// CHECK: %[[x:.*]] = modelica.alloc : !modelica.array<3x!modelica.int>
// CHECK: %[[y:.*]] = modelica.call @callee(%[[x]]) : (!modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK: return %[[x]], %[[y]]

func.func @caller() -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0:2 = modelica.call @callee() : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>)
    return %0#0, %0#1 : !modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>
}
