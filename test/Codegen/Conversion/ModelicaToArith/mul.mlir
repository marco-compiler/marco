// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[result:.*]] = arith.muli %[[arg0_casted]], %[[arg1_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.int {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = arith.mulf %[[arg0_casted]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.real {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[arg0_f64:.*]] = arith.sitofp %[[arg0_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.mulf %[[arg0_f64]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.real {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[arg1_f64:.*]] = arith.sitofp %[[arg1_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.mulf %[[arg0_casted]], %[[arg1_f64]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.real {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// Scalar and array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg1_dim0:.*]] = modelica.dim %[[arg1]], %[[c0]]
// CHECK-DAG:   %[[arg1_dim1:.*]] = modelica.dim %[[arg1]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg1_dim1]] : <3x?x!modelica.real>
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[arg1_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[arg1_dim1]] step %[[c1]] {
// CHECK:               %[[array_value:.*]] = modelica.load %[[arg1]][%[[index_0]], %[[index_1]]]
// CHECK:               %[[array_value_casted:.*]] = builtin.unrealized_conversion_cast %[[array_value]] : !modelica.real to f64
// CHECK:               %[[mul:.*]] = arith.mulf %[[arg0_casted]], %[[array_value_casted]]
// CHECK:               %[[mul_casted:.*]] = builtin.unrealized_conversion_cast %[[mul]] : f64 to !modelica.real
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[mul_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real> {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.array<3x?x!modelica.real>) -> !modelica.array<3x?x!modelica.real>
    func.return %0 : !modelica.array<3x?x!modelica.real>
}

// -----

// 1D array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x!modelica.real>, %[[arg1:.*]]: !modelica.array<?x!modelica.real>) -> !modelica.real
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = modelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg1_dim0:.*]] = modelica.dim %[[arg1]], %[[c0]]
// CHECK-DAG:   %[[dim0_cmp:.*]] = arith.cmpi eq, %[[arg0_dim0]], %[[arg1_dim0]]
// CHECK-DAG:   cf.assert %[[dim0_cmp]]
// CHECK-DAG:   %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.for %[[index_0:.*]] = %[[c0]] to %[[arg0_dim0]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:       %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_0]]]
// CHECK-DAG:       %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:       %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:           %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:           %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:           %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:           %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !modelica.real to f64
// CHECK:           scf.yield %[[add_casted_2]] : f64
// CHECK:       }
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK:       return %[[result_casted]]

func.func @foo(%arg0 : !modelica.array<?x!modelica.real>, %arg1 : !modelica.array<?x!modelica.real>) -> !modelica.real {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.array<?x!modelica.real>, !modelica.array<?x!modelica.real>) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// 2D array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.real>, %[[arg1:.*]]: !modelica.array<?x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = modelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim0:.*]] = modelica.dim %[[arg1]], %[[c0]]
// CHECK-DAG:   %[[common_dim_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim0]]
// CHECK-DAG:   cf.assert %[[common_dim_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc : <3x5x!modelica.real>
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[arg0_dim0]] step %[[c1]] {
// CHECK:           %[[arg1_dim1:.*]] = modelica.dim %[[arg1]], %[[c1]]
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[arg1_dim1]] step %[[c1]] {
// CHECK:               %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:               %[[cross_product:.*]] = scf.for %[[index_2:.*]] = %[[c0]] to %[[arg0_dim1]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:               %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_2]]]
// CHECK-DAG:               %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_2]], %[[index_1]]]
// CHECK-DAG:               %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:               %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:                   %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:                   %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:                   %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:                   %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !modelica.real to f64
// CHECK:                   scf.yield %[[add_casted_2]] : f64
// CHECK:               }
// CHECK:               %[[cross_product_casted:.*]] = builtin.unrealized_conversion_cast %[[cross_product]] : f64 to !modelica.real
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[cross_product_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.real>, %arg1 : !modelica.array<?x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real> {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.array<3x?x!modelica.real>, !modelica.array<?x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>
    func.return %0 : !modelica.array<3x5x!modelica.real>
}

// -----

// 1D array and 2D array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x!modelica.real>, %[[arg1:.*]]: !modelica.array<?x3x!modelica.real>) -> !modelica.array<3x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = modelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg1_dim0:.*]] = modelica.dim %[[arg1]], %[[c0]]
// CHECK-DAG:   %[[common_dim_cmp:.*]] = arith.cmpi eq, %[[arg0_dim0]], %[[arg1_dim0]]
// CHECK-DAG:   cf.assert %[[common_dim_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc : <3x!modelica.real>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK-DAG:       %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[cross_product:.*]] = scf.for %[[index_1:.*]] = %[[c0]] to %[[arg0_dim0]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:           %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_1]], %[[index_0]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:               %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:               %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:               %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:               %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !modelica.real to f64
// CHECK:               scf.yield %[[add_casted_2]] : f64
// CHECK:           }
// CHECK:           %[[cross_product_casted:.*]] = builtin.unrealized_conversion_cast %[[cross_product]] : f64 to !modelica.real
// CHECK:           modelica.store %[[result]][%[[index_0]]], %[[cross_product_casted]]
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<?x!modelica.real>, %arg1 : !modelica.array<?x3x!modelica.real>) -> !modelica.array<3x!modelica.real> {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.array<?x!modelica.real>, !modelica.array<?x3x!modelica.real>) -> !modelica.array<3x!modelica.real>
    func.return %0 : !modelica.array<3x!modelica.real>
}

// -----

// 2D array and 1D array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.real>, %[[arg1:.*]]: !modelica.array<?x!modelica.real>) -> !modelica.array<3x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim0:.*]] = modelica.dim %[[arg1]], %[[c0]]
// CHECK-DAG:   %[[common_dim_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim0]]
// CHECK-DAG:   cf.assert %[[common_dim_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc : <3x!modelica.real>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK-DAG:       %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[cross_product:.*]] = scf.for %[[index_1:.*]] = %[[c0]] to %[[arg0_dim1]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:           %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_1]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:               %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:               %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:               %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:               %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !modelica.real to f64
// CHECK:               scf.yield %[[add_casted_2]] : f64
// CHECK:           }
// CHECK:           %[[cross_product_casted:.*]] = builtin.unrealized_conversion_cast %[[cross_product]] : f64 to !modelica.real
// CHECK:           modelica.store %[[result]][%[[index_0]]], %[[cross_product_casted]]
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.real>, %arg1 : !modelica.array<?x!modelica.real>) -> !modelica.array<3x!modelica.real> {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.array<3x?x!modelica.real>, !modelica.array<?x!modelica.real>) -> !modelica.array<3x!modelica.real>
    func.return %0 : !modelica.array<3x!modelica.real>
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> index
// CHECK: %[[result:.*]] = arith.muli %[[arg0]], %[[arg1]] : index
// CHECK: return %[[result]] : index

func.func @foo(%arg0 : index, %arg1 : index) -> index {
    %0 = modelica.mul %arg0, %arg1 : (index, index) -> index
    func.return %0 : index
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = arith.muli %[[arg0]], %[[arg1]] : i64
// CHECK: return %[[result]] : i64

func.func @foo(%arg0 : i64, %arg1 : i64) -> i64 {
    %0 = modelica.mul %arg0, %arg1 : (i64, i64) -> i64
    func.return %0 : i64
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = arith.mulf %[[arg0]], %[[arg1]] : f64
// CHECK: return %[[result]] : f64

func.func @foo(%arg0 : f64, %arg1 : f64) -> f64 {
    %0 = modelica.mul %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}
