// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<1>
    %result = modelica.log10 %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<10>
    %result = modelica.log10 %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<100>
    %result = modelica.log10 %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<1.0>
    %result = modelica.log10 %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<10.0>
    %result = modelica.log10 %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 2.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<100.0>
    %result = modelica.log10 %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: -1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.1>
    %result = modelica.log10 %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
