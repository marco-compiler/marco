// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Boolean

// CHECK-LABEL: @foo
// CHECK: arith.constant true

func.func @foo() {
    %0 = modelica.constant #modelica.bool<true>
    func.return
}

// -----

// Integer

// CHECK-LABEL: @foo
// CHECK: arith.constant 0 : i64

func.func @foo() {
    %0 = modelica.constant #modelica.int<0>
    func.return
}

// -----

// Real

// CHECK-LABEL: @foo
// CHECK: arith.constant 0.000000e+00 : f64

func.func @foo() {
    %0 = modelica.constant #modelica.real<0.0>
    func.return
}
