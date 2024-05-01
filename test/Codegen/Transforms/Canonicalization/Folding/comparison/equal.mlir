// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_true
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_true() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.eq %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_integerScalars_false
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_false() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.int<9>
    %result = bmodelica.eq %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_true
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_true() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.eq %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_realScalars_false
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_false() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.real<9.0>
    %result = bmodelica.eq %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_true
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_true() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.real<10.0>
    %result = bmodelica.eq %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_false
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_false() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<10>
    %y = bmodelica.constant #bmodelica.real<9.7>
    %result = bmodelica.eq %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_true
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_true() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<10.0>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.eq %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_false
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_false() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<9.7>
    %y = bmodelica.constant #bmodelica.int<10>
    %result = bmodelica.eq %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}
