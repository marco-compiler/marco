// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_firstGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_firstGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.int<9>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_integerScalars_secondGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_secondGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<9>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_firstGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_firstGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.real<9.0>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_realScalars_secondGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_secondGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<9.0>
    %y = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_firstGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_firstGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.real<9.7>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_secondGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_secondGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<9>
    %y = bmodelica.constant #bmodelica.real<9.7>
    %result = bmodelica.gte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_firstGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_firstGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<9.7>
    %y = bmodelica.constant #bmodelica.int<9>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_secondGreater
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_secondGreater() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<9.7>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.gte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}
