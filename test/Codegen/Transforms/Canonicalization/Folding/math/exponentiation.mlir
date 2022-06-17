// RUN: modelica-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<9>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.pow %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_realScalars
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.pow %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars1
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars1() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.pow %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars2
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars2() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.pow %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}
