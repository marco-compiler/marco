// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_firstSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<9>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_integerScalars_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_secondSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<9>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_firstSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.0>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_realScalars_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_secondSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<9.0>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_firstSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<10.2>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_secondSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<9.7>
    %result = modelica.lte %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_firstSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.7>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_equal() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_secondSmaller() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.2>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.lte %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}
