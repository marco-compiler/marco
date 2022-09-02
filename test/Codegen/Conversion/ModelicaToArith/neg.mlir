// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Integer operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int) -> !modelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.int) -> !modelica.int {
    %0 = modelica.neg %arg0 : !modelica.int -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// Real operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real) -> !modelica.real
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK: %[[result:.*]] = arith.negf %[[arg0_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.real) -> !modelica.real {
    %0 = modelica.neg %arg0 : !modelica.real -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Integer array operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.int>) -> !modelica.array<3x?x!modelica.int>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg0_dim1]] : !modelica.array<3x?x!modelica.int>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[zero:.*]] = arith.constant 0 : i64
// CHECK-DAG:           %[[operand:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[operand_casted:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !modelica.int to i64
// CHECK:               %[[neg:.*]] = arith.subi %[[zero]], %[[operand_casted]]
// CHECK:               %[[neg_casted:.*]] = builtin.unrealized_conversion_cast %[[neg]] : i64 to !modelica.int
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[neg_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.int>) -> !modelica.array<3x?x!modelica.int> {
    %0 = modelica.neg %arg0 : !modelica.array<3x?x!modelica.int> -> !modelica.array<3x?x!modelica.int>
    func.return %0 : !modelica.array<3x?x!modelica.int>
}

// -----

// Real array operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg0_dim1]] : !modelica.array<3x?x!modelica.real>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[operand:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[operand_casted:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !modelica.real to f64
// CHECK:               %[[neg:.*]] = arith.negf %[[operand_casted]] : f64
// CHECK:               %[[neg_casted:.*]] = builtin.unrealized_conversion_cast %[[neg]] : f64 to !modelica.real
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[neg_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real> {
    %0 = modelica.neg %arg0 : !modelica.array<3x?x!modelica.real> -> !modelica.array<3x?x!modelica.real>
    func.return %0 : !modelica.array<3x?x!modelica.real>
}
