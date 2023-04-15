// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=true})" | FileCheck --check-prefix="CHECK-CALLEE" %s
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=true})" | FileCheck --check-prefix="CHECK-CALLEE" %s

// Static output arrays can be promoted.

// CHECK-CALLEE: modelica.raw_function @callee
// CHECK-CALLEE-SAME: (%[[x:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK-CALLEE: %[[y:.*]] = modelica.raw_variable %{{.*}} : !modelica.variable<?x!modelica.int, output>
// CHECK-CALLEE: %[[y_value:.*]] = modelica.raw_variable_get %[[y]]
// CHECK-CALLEE: modelica.raw_return %[[y_value]]

modelica.function @callee {
    modelica.variable @x : !modelica.variable<3x!modelica.int, output>

    modelica.variable @y : !modelica.variable<?x!modelica.int, output> [fixed] {
        %0 = arith.constant 2 : index
        modelica.yield %0 : index
    }
}

// CHECK-CALLER: func.func @caller
// CHECK-CALLER: %[[x:.*]] = modelica.alloc : !modelica.array<3x!modelica.int>
// CHECK-CALLER: %[[y:.*]] = modelica.call @callee(%[[x]]) : (!modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK-CALLER: return %[[x]], %[[y]]

func.func @caller() -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0:2 = modelica.call @callee() : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>)
    return %0#0, %0#1 : !modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>
}
