// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_true
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalars_true() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.eq %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_integerScalars_false
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_integerScalars_false() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<9>
    %result = modelica.eq %x, %y : (!modelica.int, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_true
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalars_true() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.eq %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_realScalars_false
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_realScalars_false() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.real<9.0>
    %result = modelica.eq %x, %y : (!modelica.real, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_true
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_true() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<10.0>
    %result = modelica.eq %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_false
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_false() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<9.7>
    %result = modelica.eq %x, %y : (!modelica.int, !modelica.real) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_true
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_true() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.eq %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_false
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test_mixedScalars_integerReal_false() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<9.7>
    %y = modelica.constant #modelica.int<10>
    %result = modelica.eq %x, %y : (!modelica.real, !modelica.int) -> !modelica.bool
    return %result : !modelica.bool
}
