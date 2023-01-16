// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Boolean attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant true
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i1 to !modelica.bool
// CHECK: return %[[result]]

func.func @foo() -> !modelica.bool {
    %0 = modelica.constant #modelica.bool<true>
    func.return %0 : !modelica.bool
}

// -----

// Integer attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i64 to !modelica.int
// CHECK: return %[[result]]

func.func @foo() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    func.return %0 : !modelica.int
}

// -----

// Real attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f64 to !modelica.real
// CHECK: return %[[result]]

func.func @foo() -> !modelica.real {
    %0 = modelica.constant #modelica.real<0.0>
    func.return %0 : !modelica.real
}

// -----

// MLIR index attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : index
// CHECK: return %[[cst]]

func.func @foo() -> index {
    %0 = modelica.constant 0 : index
    func.return %0 : index
}

// -----

// MLIR integer attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: return %[[cst]]

func.func @foo() -> i64 {
    %0 = modelica.constant 0 : i64
    func.return %0 : i64
}

// -----

// MLIR float attribute

// CHECK-LABEL: @foo
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: return %[[cst]]

func.func @foo() -> f64 {
    %0 = modelica.constant 0.0 : f64
    func.return %0 : f64
}
