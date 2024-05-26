// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// Boolean attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant true
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i1 to !bmodelica.bool
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.bool {
    %0 = bmodelica.constant #bmodelica<bool true>
    func.return %0 : !bmodelica.bool
}

// -----

// Integer attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i64 to !bmodelica.int
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica<int 0>
    func.return %0 : !bmodelica.int
}

// -----

// Real attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f64 to !bmodelica.real
// CHECK: return %[[result]]

func.func @foo() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica<real 0.0>
    func.return %0 : !bmodelica.real
}

// -----

// Integer attribute.

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[false, false, false], [true, true, true]]> : tensor<2x3xi1>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xi1> to tensor<2x3x!bmodelica.bool>
// CHECK: return %[[result]]

func.func @foo() -> tensor<2x3x!bmodelica.bool> {
    %0 = bmodelica.constant #bmodelica.dense_bool<[false, false, false, true, true, true]> : tensor<2x3x!bmodelica.bool>
    func.return %0 : tensor<2x3x!bmodelica.bool>
}

// -----

// Integer attribute.

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xi64> to tensor<2x3x!bmodelica.int>
// CHECK: return %[[result]]

func.func @foo() -> tensor<2x3x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.dense_int<[1, 2, 3, 4, 5, 6]> : tensor<2x3x!bmodelica.int>
    func.return %0 : tensor<2x3x!bmodelica.int>
}

// -----

// Real attribute.

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xf64> to tensor<2x3x!bmodelica.real>
// CHECK: return %[[result]]

func.func @foo() -> tensor<2x3x!bmodelica.real> {
    %0 = bmodelica.constant #bmodelica.dense_real<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<2x3x!bmodelica.real>
    func.return %0 : tensor<2x3x!bmodelica.real>
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
