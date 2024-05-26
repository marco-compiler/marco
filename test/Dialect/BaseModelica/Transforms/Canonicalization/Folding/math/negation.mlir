// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica<int -3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.neg %x: !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// Real operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica<real -3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.neg %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// MLIR index operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant -3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.neg %x : index -> index
    return %result : index
}

// -----

// MLIR integer operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant -3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.neg %x : i64 -> i64
    return %result : i64
}

// -----

// MLIR float operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant -3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.neg %x : f64 -> f64
    return %result : f64
}
