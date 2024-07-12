// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK-LABEL: @foo

func.func @foo() {
  bmodelica.assert {level = 2 : i64, message = "Test"} {
    %0 = arith.constant false
    bmodelica.yield %0 : i1
  }
  func.return
}
