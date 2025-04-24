// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK-LABEL: @warning

// CHECK-DAG: %[[condition:.*]] = arith.constant false
// CHECK-DAG: %[[message:.*]] = runtime.string "message"
// CHECK-DAG: %[[level:.*]] = arith.constant 0
// CHECK: runtime.call @_Massert_void_i1_pvoid_i64(%[[condition]], %[[message]], %[[level]])

func.func @warning() {
  bmodelica.assert attributes {level = 0 : i32, message = "message"} {
    %0 = arith.constant false
    bmodelica.yield %0 : i1
  }
  func.return
}

// -----

// CHECK-LABEL: @error

// CHECK-DAG: %[[condition:.*]] = arith.constant false
// CHECK-DAG: %[[message:.*]] = runtime.string "message"
// CHECK-DAG: %[[level:.*]] = arith.constant 1
// CHECK: runtime.call @_Massert_void_i1_pvoid_i64(%[[condition]], %[[message]], %[[level]])

func.func @error() {
  bmodelica.assert attributes {level = 1 : i32, message = "message"} {
    %0 = arith.constant false
    bmodelica.yield %0 : i1
  }
  func.return
}
