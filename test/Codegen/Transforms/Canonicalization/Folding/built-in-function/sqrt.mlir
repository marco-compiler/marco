// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<0>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<1>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<4>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<9>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<4.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<9.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
