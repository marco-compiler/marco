// RUN: modelica-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.int<-3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.neg %x: !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: -3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %result = modelica.neg %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
