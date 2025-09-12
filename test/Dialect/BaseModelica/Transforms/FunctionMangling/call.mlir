// RUN: modelica-opt %s --split-input-file --function-mangling | FileCheck %s

// CHECK-LABEL: @test
// CHECK:       bmodelica.call @_Mfoo() : () -> ()

func.func @test() {
  bmodelica.call @foo() : () -> ()
  func.return
}

bmodelica.function @foo {

}
