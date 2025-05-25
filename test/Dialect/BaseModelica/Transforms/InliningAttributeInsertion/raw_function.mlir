// RUN: modelica-opt %s --split-input-file --inlining-attr-insertion | FileCheck %s

// CHECK-LABEL: @inlineTrue
// CHECK: bmodelica.call @inlineTrue() {inline_hint} : () -> ()

bmodelica.raw_function @inlineTrue() attributes {inline = true} {
    bmodelica.raw_return
}

func.func @foo() {
    bmodelica.call @inlineTrue() : () -> ()
    func.return
}

// -----

// CHECK-LABEL: @inlineFalse
// CHECK: bmodelica.call @inlineFalse() : () -> ()

bmodelica.raw_function @inlineFalse() attributes {inline = false} {
    bmodelica.raw_return
}

func.func @foo() {
    bmodelica.call @inlineFalse() : () -> ()
    func.return
}

// -----

// CHECK-LABEL: @default
// CHECK: bmodelica.call @default() : () -> ()

bmodelica.raw_function @default() {
    bmodelica.raw_return
}

func.func @foo() {
    bmodelica.call @default() : () -> ()
    func.return
}
