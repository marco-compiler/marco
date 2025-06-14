// RUN: modelica-opt %s --split-input-file --inlining-attr-insertion | FileCheck %s

// CHECK-LABEL: @inlineTrue
// CHECK: bmodelica.call @inlineTrue() {inline_hint} : () -> ()

bmodelica.function @inlineTrue attributes {inline = true} {

}

func.func @foo() {
    bmodelica.call @inlineTrue() : () -> ()
    func.return
}

// -----

// CHECK-LABEL: @inlineFalse
// CHECK: bmodelica.call @inlineFalse() : () -> ()

bmodelica.function @inlineFalse attributes {inline = false} {

}

func.func @foo() {
    bmodelica.call @inlineFalse() : () -> ()
    func.return
}

// -----

// CHECK-LABEL: @default
// CHECK: bmodelica.call @default() : () -> ()

bmodelica.function @default {

}

func.func @foo() {
    bmodelica.call @default() : () -> ()
    func.return
}
