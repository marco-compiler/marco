// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.add %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// Real operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.add %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// Integer and real operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.add %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// Real and integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.add %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR index operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 5 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3 : index
    %y = modelica.constant 2 : index
    %result = modelica.add %x, %y : (index, index) -> index
    return %result : index
}

// -----

// MLIR integer operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 5 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3 : i64
    %y = modelica.constant 2 : i64
    %result = modelica.add %x, %y : (i64, i64) -> i64
    return %result : i64
}

// -----

// MLIR float operands.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 5.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3.0 : f64
    %y = modelica.constant 2.0 : f64
    %result = modelica.add %x, %y : (f64, f64) -> f64
    return %result : f64
}

// -----

// Integer range and Integer offset.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int_range<7, 9, 1>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica<range !modelica.int>) {
    %x = modelica.constant #modelica.int_range<5, 7, 1>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.add %x, %y : (!modelica<range !modelica.int>, !modelica.int) -> !modelica<range !modelica.int>
    return %result : !modelica<range !modelica.int>
}

// -----

// Integer range and Real offset.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real_range<7.000000e+00, 9.000000e+00, 1.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica<range !modelica.real>) {
    %x = modelica.constant #modelica.int_range<5, 7, 1>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.add %x, %y : (!modelica<range !modelica.int>, !modelica.real) -> !modelica<range !modelica.real>
    return %result : !modelica<range !modelica.real>
}

// -----

// Real range and Integer offset.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real_range<7.000000e+00, 9.000000e+00, 1.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica<range !modelica.real>) {
    %x = modelica.constant #modelica.real_range<5.0, 7.0, 1.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.add %x, %y : (!modelica<range !modelica.real>, !modelica.int) -> !modelica<range !modelica.real>
    return %result : !modelica<range !modelica.real>
}

// -----

// Real range and Real offset.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real_range<7.000000e+00, 9.000000e+00, 1.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica<range !modelica.real>) {
    %x = modelica.constant #modelica.real_range<5.0, 7.0, 1.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.add %x, %y : (!modelica<range !modelica.real>, !modelica.real) -> !modelica<range !modelica.real>
    return %result : !modelica<range !modelica.real>
}
