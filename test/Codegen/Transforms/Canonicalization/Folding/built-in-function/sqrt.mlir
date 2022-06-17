// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<0>
    %result = modelica.sqrt %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<1>
    %result = modelica.sqrt %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<4>
    %result = modelica.sqrt %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<9>
    %result = modelica.sqrt %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.0>
    %result = modelica.sqrt %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<1.0>
    %result = modelica.sqrt %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 2.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<4.0>
    %result = modelica.sqrt %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<9.0>
    %result = modelica.sqrt %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
