// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_first
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_first() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_integerScalars_second
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_second() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<2>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_equal() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test_realScalars_first
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_first() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_realScalars_second
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_second() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<2.0>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_equal() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_first
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_first() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars_integerReal_second
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_second() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<2>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_equal() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.max %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_first
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_first() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars_realInteger_second
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_second() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<2.0>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_equal() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.max %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}
