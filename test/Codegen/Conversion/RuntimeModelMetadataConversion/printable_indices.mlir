// RUN: modelica-opt %s --split-input-file --convert-runtime-model-metadata | FileCheck %s

// COM: Non-printable scalar variable.

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

runtime.printable_indices [false]

// -----

// COM: Printable scalar variable.

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

runtime.printable_indices [true]

// -----

// COM: Non-printable array variable.

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

runtime.printable_indices [{}]

// -----

// COM: Printable array variable.

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

runtime.printable_indices [{[0,2]}]
