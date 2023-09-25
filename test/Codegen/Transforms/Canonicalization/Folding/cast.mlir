// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Boolean -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// Boolean -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> !modelica.int
    return %result : !modelica.int
}

// -----

// Boolean -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> !modelica.real
    return %result : !modelica.real
}

// -----

// Boolean -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 1 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> index
    return %result : index
}

// -----

// Boolean -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 1 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> i64
    return %result : i64
}

// -----

// Boolean -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 1.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.cast %x: !modelica.bool -> f64
    return %result : f64
}

// -----

// Integer -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> !modelica.int
    return %result : !modelica.int
}

// -----

// Integer -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// Integer -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> !modelica.real
    return %result : !modelica.real
}

// -----

// Integer -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> index
    return %result : index
}

// -----

// Integer -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> i64
    return %result : i64
}

// -----

// Integer -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant #modelica.int<3>
    %result = modelica.cast %x: !modelica.int -> f64
    return %result : f64
}

// -----

// Real -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %result = modelica.cast %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// Real -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.real<3.0>
    %result = modelica.cast %x : !modelica.real -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// Real -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.real<3.5>
    %result = modelica.cast %x : !modelica.real -> !modelica.int
    return %result : !modelica.int
}

// -----

// Real -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant #modelica.real<3.5>
    %result = modelica.cast %x : !modelica.real -> index
    return %result : index
}

// -----

// Real -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant #modelica.real<3.5>
    %result = modelica.cast %x : !modelica.real -> i64
    return %result : i64
}

// -----

// MLIR index -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> index
    return %result : index
}

// -----

// MLIR index -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// MLIR index -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> !modelica.int
    return %result : !modelica.int
}

// -----

// MLIR index -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR index -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> i64
    return %result : i64
}

// -----

// MLIR index -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3 : index
    %result = modelica.cast %x : index -> f64
    return %result : f64
}

// -----

// MLIR integer -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> i64
    return %result : i64
}

// -----

// MLIR integer -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// MLIR integer -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> !modelica.int
    return %result : !modelica.int
}

// -----

// MLIR integer -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR integer -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> index
    return %result : index
}

// -----

// MLIR integer -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3 : i64
    %result = modelica.cast %x : i64 -> f64
    return %result : f64
}

// -----

// MLIR float -> MLIR float

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3.000000e+00 : f64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (f64) {
    %x = modelica.constant 3.0 : f64
    %result = modelica.cast %x : f64 -> f64
    return %result : f64
}

// -----

// MLIR float -> Boolean

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant 3.0 : f64
    %result = modelica.cast %x : f64 -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// MLIR float -> Integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant 3.5 : f64
    %result = modelica.cast %x : f64 -> !modelica.int
    return %result : !modelica.int
}

// -----

// MLIR float -> Real

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant #modelica.real<3.500000e+00>
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant 3.5 : f64
    %result = modelica.cast %x : f64 -> !modelica.real
    return %result : !modelica.real
}

// -----

// MLIR float -> MLIR index

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : index
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (index) {
    %x = modelica.constant 3.5 : f64
    %result = modelica.cast %x : f64 -> index
    return %result : index
}

// -----

// MLIR float -> MLIR integer

// CHECK-LABEL: @test
// CHECK-NEXT: %[[cst:.*]] = modelica.constant 3 : i64
// CHECK-NEXT: return %[[cst]]

func.func @test() -> (i64) {
    %x = modelica.constant 3.5 : f64
    %result = modelica.cast %x : f64 -> i64
    return %result : i64
}

