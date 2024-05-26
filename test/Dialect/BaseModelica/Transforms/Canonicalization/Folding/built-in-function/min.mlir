// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars_first
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_first() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// CHECK-LABEL: @test_integerScalars_second
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_second() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// CHECK-LABEL: @test_integerScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars_equal() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test_realScalars_first
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_first() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_realScalars_second
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_second() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_realScalars_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars_equal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_mixedScalars_integerReal_first
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_first() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_mixedScalars_integerReal_second
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_second() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_mixedScalars_integerReal_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_integerReal_equal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test_mixedScalars_realInteger_first
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_first() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_mixedScalars_realInteger_second
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_second() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
}

// CHECK-LABEL: @test_mixedScalars_realInteger_equal
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars_realInteger_equal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
}
