// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<-3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.neg %x: !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// Real operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<-3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %result = modelica.neg %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR index operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant -3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3 : index
    %result = modelica.neg %x : index -> index
    return %result : index
}

// -----

// MLIR integer operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant -3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3 : i64
    %result = modelica.neg %x : i64 -> i64
    return %result : i64
}

// -----

// MLIR float operand.

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant -3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3.0 : f64
    %result = modelica.neg %x : f64 -> f64
    return %result : f64
}
