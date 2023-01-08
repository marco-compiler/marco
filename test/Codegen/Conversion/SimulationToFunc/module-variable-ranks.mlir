// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-UNKNOWN"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-0"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-1"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-2"

// CHECK-VAR-UNKNOWN-LABEL: @getVariableRank
// CHECK-VAR-UNKNOWN-SAME:  (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-UNKNOWN-DAG:   %[[default:.*]] = arith.constant 0 : i64
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
// CHECK-VAR-0-DAG:         %[[rank:.*]] = arith.constant 0 : i64
// CHECK-VAR-0:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-0:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-0:             return %[[result]]

// CHECK-VAR-1-LABEL:   @getVariableRank
// CHECK-VAR-1-SAME:    (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-1:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-1:             1: ^[[var_block:.*]],
// CHECK-VAR-1:         ]
// CHECK-VAR-1:         ^[[var_block]]:
// CHECK-VAR-1-DAG:         %[[rank:.*]] = arith.constant 1 : i64
// CHECK-VAR-1:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-1:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-1:             return %[[result]]

// CHECK-VAR-2-LABEL:   @getVariableRank
// CHECK-VAR-2-SAME:    (%[[varNumber:.*]]: i64) -> i64
// CHECK-VAR-2:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-2:             1: ^[[var_block:.*]],
// CHECK-VAR-2:         ]
// CHECK-VAR-2:         ^[[var_block]]:
// CHECK-VAR-2-DAG:         %[[rank:.*]] = arith.constant 2 : i64
// CHECK-VAR-2:             cf.br ^[[outBlock:.*]](%[[rank]] : i64)
// CHECK-VAR-2:         ^[[outBlock]](%[[result:.*]]: i64):
// CHECK-VAR-2:             return %[[result]]

#var0 = #simulation.variable<name = "var0">
#var1 = #simulation.variable<name = "var1", dimensions = [3]>
#var2 = #simulation.variable<name = "var2", dimensions = [3, 5]>

simulation.module attributes {variables = [#var0, #var1, #var2]} {

}
