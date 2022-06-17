// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<-2>
    %result = modelica.abs %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<0>
    %result = modelica.abs %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_integerScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalar() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<2>
    %result = modelica.abs %x : !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<1.500000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<-1.5>
    %result = modelica.abs %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.0>
    %result = modelica.abs %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_realScalar
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<1.500000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalar() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<1.5>
    %result = modelica.abs %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
