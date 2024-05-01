// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<6>
    %y = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.rem %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<8>
    %y = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.rem %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.int<-3>
    %result = bmodelica.rem %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.int<-1>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<-10>
    %y = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.rem %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<6.0>
    %y = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<8.5>
    %y = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<1.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<10.5>
    %y = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<-1.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<-10.5>
    %y = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<1.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<10.5>
    %y = bmodelica.constant #bmodelica.real<-3.0>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<8.5>
    %y = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.rem %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.real<3.75>
    %result = bmodelica.rem %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}
