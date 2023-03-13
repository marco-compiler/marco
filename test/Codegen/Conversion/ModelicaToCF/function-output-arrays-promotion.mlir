// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=true})" | FileCheck %s

// Static output arrays can be promoted.

// CHECK-LABEL: @callee
// CHECK-SAME: (%[[x:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK: %[[y:.*]] = modelica.raw_variable %{{.*}} : !modelica.member<?x!modelica.int, output>
// CHECK: %[[y_value:.*]] = modelica.raw_variable_get %[[y]]
// CHECK: modelica.raw_return %[[y_value]]

modelica.function @callee {
    modelica.variable @x : !modelica.member<3x!modelica.int, output>

    modelica.variable @y : !modelica.member<?x!modelica.int, output> [fixed] {
        %0 = arith.constant 2 : index
        modelica.yield %0 : index
    }
}

// CHECK-LABEL: @caller
// CHECK: %[[x:.*]] = modelica.alloc : !modelica.array<3x!modelica.int>
// CHECK: %[[y:.*]] = modelica.call @callee(%[[x]]) : (!modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK: return %[[x]], %[[y]]

func.func @caller() -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0:2 = modelica.call @callee() : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>)
    return %0#0, %0#1 : !modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>
}