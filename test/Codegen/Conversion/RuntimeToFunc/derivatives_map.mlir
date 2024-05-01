// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// No derivatives

// CHECK:       func.func @getDerivative(%[[variable:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[default]] : i64),
// CHECK-DAG:           0: ^[[var0_block:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:  ^[[var0_block]]:
// CHECK-NEXT:      %[[var0_der:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.br ^[[out]](%[[var0_der]] : i64)
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

runtime.derivatives_map [-1]

// -----

// CHECK:       func.func @getDerivative(%[[variable:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[default]] : i64),
// CHECK-DAG:           0: ^[[var0_block:.*]],
// CHECK-DAG:           1: ^[[var1_block:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:  ^[[var0_block]]:
// CHECK-NEXT:      %[[var0_der:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      cf.br ^[[out]](%[[var0_der]] : i64)
// CHECK-NEXT:  ^[[var1_block]]:
// CHECK-NEXT:      %[[var1_der:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.br ^[[out]](%[[var1_der]] : i64)
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

runtime.derivatives_map [1, -1]
