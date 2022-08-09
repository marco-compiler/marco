// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @boolean
// CHECK: arith.constant true

func.func @boolean() {
    %0 = modelica.constant #modelica.bool<true>
    func.return
}

// -----

// CHECK-LABEL: @integer
// CHECK: arith.constant 0 : i64

func.func @integer() {
    %0 = modelica.constant #modelica.int<0>
    func.return
}

// -----

// CHECK-LABEL: @real
// CHECK: arith.constant 0.000000e+00 : f64

func.func @real() {
    %0 = modelica.constant #modelica.real<0.0>
    func.return
}
