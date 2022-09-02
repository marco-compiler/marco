// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[result:.*]] = arith.addi %[[arg0_casted]], %[[arg1_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = arith.addf %[[arg0_casted]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.real {
    %0 = modelica.add %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted_1:.*]] = modelica.cast %[[arg0]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[arg0_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg0_casted_1]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = arith.addf %[[arg0_casted_2]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.real {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted_1:.*]] = modelica.cast %[[arg1]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[arg1_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg1_casted_1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = arith.addf %[[arg0_casted]], %[[arg1_casted_2]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.real {
    %0 = modelica.add %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Integer array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.int>, %[[arg1:.*]]: !modelica.array<3x?x!modelica.int>) -> !modelica.array<3x?x!modelica.int>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim1:.*]] = modelica.dim %[[arg1]], %[[c1]]
// CHECK-DAG:   %[[dim1_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim1]]
// CHECK-DAG:   cf.assert %[[dim1_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg0_dim1]] : !modelica.array<3x?x!modelica.int>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.int to i64
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.int to i64
// CHECK:               %[[add:.*]] = arith.addi %[[lhs_casted]], %[[rhs_casted]] : i64
// CHECK:               %[[add_casted:.*]] = builtin.unrealized_conversion_cast %[[add]] : i64 to !modelica.int
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[add_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.int>, %arg1 : !modelica.array<3x?x!modelica.int>) -> !modelica.array<3x?x!modelica.int> {
    %0 = modelica.add %arg0, %arg1 : (!modelica.array<3x?x!modelica.int>, !modelica.array<3x?x!modelica.int>) -> !modelica.array<3x?x!modelica.int>
    func.return %0 : !modelica.array<3x?x!modelica.int>
}

// -----

// Real array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.real>, %[[arg1:.*]]: !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim1:.*]] = modelica.dim %[[arg1]], %[[c1]]
// CHECK-DAG:   %[[dim1_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim1]]
// CHECK-DAG:   cf.assert %[[dim1_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg0_dim1]] : !modelica.array<3x?x!modelica.real>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:               %[[add:.*]] = arith.addf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:               %[[add_casted:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[add_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.real>, %arg1 : !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real> {
    %0 = modelica.add %arg0, %arg1 : (!modelica.array<3x?x!modelica.real>, !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real>
    func.return %0 : !modelica.array<3x?x!modelica.real>
}
