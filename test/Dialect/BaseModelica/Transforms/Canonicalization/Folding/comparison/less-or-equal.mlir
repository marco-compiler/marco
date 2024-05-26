// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_firstSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 9>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_integerScalars_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_secondSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 9>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_realScalars_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_firstSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 9.0>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_realScalars_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_secondSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 9.0>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_firstSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 10.2>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_integerReal_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_secondSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 9.7>
    %result = bmodelica.lte %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_firstSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_firstSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 9.7>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_equal() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// CHECK-LABEL: @test_mixedScalars_realInteger_secondSmaller
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_secondSmaller() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.2>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lte %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
}
