// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// Boolean operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK-DAG: %[[zero:.*]] = arith.constant false
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.bool -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Integer operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.int -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x]], %[[zero]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.real -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Boolean array operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x?x!bmodelica.bool>) -> !bmodelica.array<3x?x!bmodelica.bool>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.alloc %[[arg0_dim1]] : <3x?x!bmodelica.bool>
// CHECK-DAG:   %[[result_dim0:.*]] = bmodelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = bmodelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[operand:.*]] = bmodelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[operand_casted:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !bmodelica.bool to i1
// CHECK-DAG:           %[[eq:.*]] = arith.cmpi eq, %[[operand_casted]], %[[false]] : i1
// CHECK:               %[[eq_casted:.*]] = builtin.unrealized_conversion_cast %[[eq]] : i1 to !bmodelica.bool
// CHECK:               bmodelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[eq_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<3x?x!bmodelica.bool>) -> !bmodelica.array<3x?x!bmodelica.bool> {
    %0 = bmodelica.not %arg0 : !bmodelica.array<3x?x!bmodelica.bool> -> !bmodelica.array<3x?x!bmodelica.bool>
    func.return %0 : !bmodelica.array<3x?x!bmodelica.bool>
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[zero]] : index
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : index) -> i1 {
    %0 = bmodelica.not %arg0 : index -> i1
    func.return %0 : i1
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[zero]] : i64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : i64) -> i1 {
    %0 = bmodelica.not %arg0 : i64 -> i1
    func.return %0 : i1
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = arith.cmpf oeq, %[[arg0]], %[[zero]] : f64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : f64) -> i1 {
    %0 = bmodelica.not %arg0 : f64 -> i1
    func.return %0 : i1
}
