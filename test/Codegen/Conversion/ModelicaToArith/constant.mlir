// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Boolean attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant true
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i1 to !bmodelica.bool
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.bool {
    %0 = bmodelica.constant #bmodelica.bool<true>
    func.return %0 : !bmodelica.bool
}

// -----

// Integer attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i64 to !bmodelica.int
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica.int<0>
    func.return %0 : !bmodelica.int
}

// -----

// Real attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f64 to !bmodelica.real
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica.real<0.0>
    func.return %0 : !bmodelica.real
}

// -----

// MLIR index attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : index
// CHECK: return %[[cst]]

func.func @foo() -> index {
    %0 = bmodelica.constant 0 : index
    func.return %0 : index
}

// -----

// MLIR integer attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: return %[[cst]]

func.func @foo() -> i64 {
    %0 = bmodelica.constant 0 : i64
    func.return %0 : i64
}

// -----

// MLIR float attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: return %[[cst]]

func.func @foo() -> f64 {
    %0 = bmodelica.constant 0.0 : f64
    func.return %0 : f64
}
