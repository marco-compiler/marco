// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Boolean -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// Boolean -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// Boolean -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// Boolean -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 1 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> index
    return %result : index
}

// -----

// Boolean -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 1 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> i64
    return %result : i64
}

// -----

// Boolean -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 1.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant #bmodelica.bool<true>
    %result = bmodelica.cast %x: !bmodelica.bool -> f64
    return %result : f64
}

// -----

// Integer -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// Integer -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// Integer -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// Integer -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> index
    return %result : index
}

// -----

// Integer -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> i64
    return %result : i64
}

// -----

// Integer -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant #bmodelica.int<3>
    %result = bmodelica.cast %x: !bmodelica.int -> f64
    return %result : f64
}

// -----

// Real -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// Real -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica.real<3.0>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// Real -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.real<3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// Real -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant #bmodelica.real<3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> index
    return %result : index
}

// -----

// Real -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant #bmodelica.real<3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> i64
    return %result : i64
}

// -----

// MLIR index -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> index
    return %result : index
}

// -----

// MLIR index -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// MLIR index -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// MLIR index -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// MLIR index -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> i64
    return %result : i64
}

// -----

// MLIR index -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> f64
    return %result : f64
}

// -----

// MLIR integer -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> i64
    return %result : i64
}

// -----

// MLIR integer -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// MLIR integer -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// MLIR integer -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// MLIR integer -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> index
    return %result : index
}

// -----

// MLIR integer -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> f64
    return %result : f64
}

// -----

// MLIR float -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.cast %x : f64 -> f64
    return %result : f64
}

// -----

// MLIR float -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// MLIR float -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// MLIR float -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.500000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// MLIR float -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> index
    return %result : index
}

// -----

// MLIR float -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = bmodelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> i64
    return %result : i64
}

