// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK-LABEL: @foo

// CHECK: runtime.string {string = "Test"} : !runtime.string
// CHECK: runtime.call @_Massert_void_i1_pvoid_i64

func.func @foo() {
  bmodelica.assert {level = 1 : i64, message = "Test"} {
    %0 = arith.constant false
    bmodelica.yield %0 : i1
  }
  func.return
}
