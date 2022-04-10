// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_firstGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalars_firstGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<9>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalars_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_integerScalars_secondGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalars_secondGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<9>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_firstGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalars_firstGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<9.0>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalars_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_realScalars_secondGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalars_secondGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.0>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_firstGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_firstGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<9.7>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_secondGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_secondGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<9>
    %y = modelica.constant #modelica.real<9.7>
    %result = modelica.gt %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_firstGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_realInteger_firstGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.7>
    %y = modelica.constant #modelica.int<9>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_realInteger_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_secondGreater
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_realInteger_secondGreater() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.7>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.gt %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}
