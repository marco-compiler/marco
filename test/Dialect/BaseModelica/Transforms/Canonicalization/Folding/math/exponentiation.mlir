// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<int 9>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.pow %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// Real operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 9.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.pow %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// Integer base and real exponent.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 9.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.pow %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// Real base and integer exponent.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<real 9.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.pow %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// MLIR index operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 9 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant 3 : index
    %y = bmodelica.constant 2 : index
    %result = bmodelica.pow %x, %y : (index, index) -> index
    return %result : index
}

// -----

// MLIR integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 9 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %y = bmodelica.constant 2 : i64
    %result = bmodelica.pow %x, %y : (i64, i64) -> i64
    return %result : i64
}

// -----

// MLIR float operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 9.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %y = bmodelica.constant 2.0 : f64
    %result = bmodelica.pow %x, %y : (f64, f64) -> f64
    return %result : f64
}
