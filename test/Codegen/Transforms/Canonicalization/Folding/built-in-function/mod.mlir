// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.int) {
    %dividend = modelica.constant #modelica.int<8>
    %divisor = modelica.constant #modelica.int<3>
    %result = modelica.mod %dividend, %divisor : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %dividend = modelica.constant #modelica.real<8.5>
    %divisor = modelica.constant #modelica.real<3.0>
    %result = modelica.mod %dividend, %divisor : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %dividend = modelica.constant #modelica.real<8.5>
    %divisor = modelica.constant #modelica.int<3>
    %result = modelica.mod %dividend, %divisor : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.500000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %dividend = modelica.constant #modelica.int<10>
    %divisor = modelica.constant #modelica.real<3.75>
    %result = modelica.mod %dividend, %divisor : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}
