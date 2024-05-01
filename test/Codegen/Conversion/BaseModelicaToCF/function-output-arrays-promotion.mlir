// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-bmodelica-to-cf{output-arrays-promotion=true})" | FileCheck --check-prefix="CHECK-CALLEE" %s
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-bmodelica-to-cf{output-arrays-promotion=true})" | FileCheck --check-prefix="CHECK-CALLEE" %s

// Static output arrays can be promoted.

// CHECK-CALLEE: bmodelica.raw_function @callee
// CHECK-CALLEE-SAME: (%[[x:.*]]: !bmodelica.array<3x!bmodelica.int>) -> !bmodelica.array<?x!bmodelica.int>
// CHECK-CALLEE: %[[y:.*]] = bmodelica.raw_variable %{{.*}} : !bmodelica.variable<?x!bmodelica.int, output>
// CHECK-CALLEE: %[[y_value:.*]] = bmodelica.raw_variable_get %[[y]]
// CHECK-CALLEE: bmodelica.raw_return %[[y_value]]

bmodelica.function @callee {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, output>

    bmodelica.variable @y : !bmodelica.variable<?x!bmodelica.int, output> [fixed] {
        %0 = arith.constant 2 : index
        bmodelica.yield %0 : index
    }
}

// CHECK-CALLER: func.func @caller
// CHECK-CALLER: %[[x:.*]] = bmodelica.alloc : <3x!bmodelica.int>
// CHECK-CALLER: %[[y:.*]] = bmodelica.call @callee(%[[x]]) : (!bmodelica.array<3x!bmodelica.int>) -> !bmodelica.array<?x!bmodelica.int>
// CHECK-CALLER: return %[[x]], %[[y]]

func.func @caller() -> (!bmodelica.array<3x!bmodelica.int>, !bmodelica.array<?x!bmodelica.int>) {
    %0:2 = bmodelica.call @callee() : () -> (!bmodelica.array<3x!bmodelica.int>, !bmodelica.array<?x!bmodelica.int>)
    return %0#0, %0#1 : !bmodelica.array<3x!bmodelica.int>, !bmodelica.array<?x!bmodelica.int>
}
