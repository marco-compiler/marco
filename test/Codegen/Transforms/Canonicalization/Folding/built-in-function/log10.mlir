// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<1>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<100>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<100.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: -1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.1>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
