// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Integer operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.neg %arg0 : !bmodelica.int -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// Real operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = arith.negf %[[arg0_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.neg %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// Integer array operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x?x!bmodelica.int>) -> !bmodelica.array<3x?x!bmodelica.int>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.alloc %[[arg0_dim1]] : <3x?x!bmodelica.int>
// CHECK-DAG:   %[[result_dim0:.*]] = bmodelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = bmodelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[zero:.*]] = arith.constant 0 : i64
// CHECK-DAG:           %[[operand:.*]] = bmodelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[operand_casted:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !bmodelica.int to i64
// CHECK:               %[[neg:.*]] = arith.subi %[[zero]], %[[operand_casted]]
// CHECK:               %[[neg_casted:.*]] = builtin.unrealized_conversion_cast %[[neg]] : i64 to !bmodelica.int
// CHECK:               bmodelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[neg_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<3x?x!bmodelica.int>) -> !bmodelica.array<3x?x!bmodelica.int> {
    %0 = bmodelica.neg %arg0 : !bmodelica.array<3x?x!bmodelica.int> -> !bmodelica.array<3x?x!bmodelica.int>
    func.return %0 : !bmodelica.array<3x?x!bmodelica.int>
}

// -----

// Real array operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x?x!bmodelica.real>) -> !bmodelica.array<3x?x!bmodelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.alloc %[[arg0_dim1]] : <3x?x!bmodelica.real>
// CHECK-DAG:   %[[result_dim0:.*]] = bmodelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = bmodelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[operand:.*]] = bmodelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[operand_casted:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !bmodelica.real to f64
// CHECK:               %[[neg:.*]] = arith.negf %[[operand_casted]] : f64
// CHECK:               %[[neg_casted:.*]] = builtin.unrealized_conversion_cast %[[neg]] : f64 to !bmodelica.real
// CHECK:               bmodelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[neg_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<3x?x!bmodelica.real>) -> !bmodelica.array<3x?x!bmodelica.real> {
    %0 = bmodelica.neg %arg0 : !bmodelica.array<3x?x!bmodelica.real> -> !bmodelica.array<3x?x!bmodelica.real>
    func.return %0 : !bmodelica.array<3x?x!bmodelica.real>
}

// -----

// MLIR index operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index) -> index
// CHECK: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0]] : index
// CHECK: return %[[result]]

func.func @foo(%arg0 : index) -> index {
    %0 = bmodelica.neg %arg0 : index -> index
    func.return %0 : index
}

// -----

// MLIR integer operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0]] : i64
// CHECK: return %[[result]]

func.func @foo(%arg0 : i64) -> i64 {
    %0 = bmodelica.neg %arg0 : i64 -> i64
    func.return %0 : i64
}

// -----

// MLIR float operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = arith.negf %[[arg0]] : f64
// CHECK: return %[[result]]

func.func @foo(%arg0 : f64) -> f64 {
    %0 = bmodelica.neg %arg0 : f64 -> f64
    func.return %0 : f64
}
