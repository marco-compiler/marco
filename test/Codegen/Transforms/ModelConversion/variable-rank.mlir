// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-UNKNOWN"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-0"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-1"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-2"

// CHECK-VAR-UNKNOWN-LABEL: @getVariableRank
// CHECK-VAR-UNKNOWN-SAME:  (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-UNKNOWN-DAG:   %[[default:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-VAR-UNKNOWN:       cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-UNKNOWN:           default: ^[[outBlock:.*]](%[[default]] : i64)
// CHECK-VAR-UNKNOWN:       ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-UNKNOWN:           return %[[result]]

// CHECK-VAR-0-LABEL:   @getVariableRank
// CHECK-VAR-0-SAME:    (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-0:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-0:             0: ^[[var_block:.*]],
// CHECK-VAR-0:         ]
// CHECK-VAR-0:         ^[[var_block]]:
// CHECK-VAR-0-DAG:         %[[rank:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-VAR-0:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-0:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-0:             return %[[result]]

// CHECK-VAR-1-LABEL:   @getVariableRank
// CHECK-VAR-1-SAME:    (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-1:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-1:             1: ^[[var_block:.*]],
// CHECK-VAR-1:         ]
// CHECK-VAR-1:         ^[[var_block]]:
// CHECK-VAR-1-DAG:         %[[rank:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-VAR-1:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-1:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-1:             return %[[result]]

// CHECK-VAR-2-LABEL:   @getVariableRank
// CHECK-VAR-2-SAME:    (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-2:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-2:             1: ^[[var_block:.*]],
// CHECK-VAR-2:         ]
// CHECK-VAR-2:         ^[[var_block]]:
// CHECK-VAR-2-DAG:         %[[rank:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-VAR-2:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-2:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-2:             return %[[result]]

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    %1 = modelica.member_create @y : !modelica.member<3x!modelica.real>
    %2 = modelica.member_create @z : !modelica.member<3x5x!modelica.real>
    modelica.yield %0, %1, %2 : !modelica.member<!modelica.real>, !modelica.member<3x!modelica.real>, !modelica.member<3x5x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<3x!modelica.real>, %arg2: !modelica.array<3x5x!modelica.real>):

}
