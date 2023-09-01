// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// CHECK:        func.func @getVariableNumOfPrintableRanges(%arg0: i64) -> i64 {
// CHECK-NEXT:      %[[zero:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      cf.switch %arg0 : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[zero]] : i64)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// Non-printable scalar variable.

simulation.printable_indices [false]

// -----

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1),
// CHECK-DAG:           0: ^[[printable:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[printable]]:
// CHECK-NEXT:          %[[true:.*]] = arith.constant true
// CHECK-NEXT:          cf.br ^[[out]](%[[true]] : i1)
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// CHECK:        func.func @getVariableNumOfPrintableRanges(%arg0: i64) -> i64 {
// CHECK-NEXT:      %[[zero:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      cf.switch %arg0 : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[zero]] : i64)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// Printable scalar variable.

simulation.printable_indices [true]

// -----

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// CHECK:        func.func @getVariableNumOfPrintableRanges(%arg0: i64) -> i64 {
// CHECK-NEXT:      %[[zero:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      cf.switch %arg0 : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[zero]] : i64)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// Non-printable array variable.

#index_set = #modeling<index_set {}>
simulation.printable_indices [#index_set]

// -----

// Printable array variable.

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1),
// CHECK-DAG:           0: ^[[printable:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[printable]]:
// CHECK-NEXT:          %[[true:.*]] = arith.constant true
// CHECK-NEXT:          cf.br ^[[out]](%[[true]] : i1)
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

// CHECK:        func.func @getVariableNumOfPrintableRanges(%arg0: i64) -> i64 {
// CHECK-NEXT:      %[[zero:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      cf.switch %arg0 : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[zero]] : i64),
// CHECK-DAG:           0: ^[[one_block:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[one_block]]:
// CHECK-NEXT:          %[[one:.*]] = arith.constant 1 : i64
// CHECK-NEXT:          cf.br ^[[out]](%[[one]] : i64)
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

#index_set = #modeling<index_set {[0,2]}>
simulation.printable_indices [#index_set]
