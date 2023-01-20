// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.int<9>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.pow %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// Real operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.pow %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// Integer base and real exponent.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars1() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.pow %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// Real base and integer exponent.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 9.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars2() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.pow %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR index operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 9 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3 : index
    %y = modelica.constant 2 : index
    %result = modelica.pow %x, %y : (index, index) -> index
    return %result : index
}

// -----

// MLIR integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 9 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3 : i64
    %y = modelica.constant 2 : i64
    %result = modelica.pow %x, %y : (i64, i64) -> i64
    return %result : i64
}

// -----

// MLIR float operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 9.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3.0 : f64
    %y = modelica.constant 2.0 : f64
    %result = modelica.pow %x, %y : (f64, f64) -> f64
    return %result : f64
}
