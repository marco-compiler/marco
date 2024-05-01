// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// Boolean operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool, %[[arg1:.*]]: !bmodelica.bool) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.bool to i1
// CHECK-DAG: %[[false:.*]] = arith.constant false
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[false]] : i1
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[false]] : i1
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.bool, %arg1 : !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[zero]] : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[zero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[zero]] : f64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[zero]] : f64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[lhsZero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[lhsZero]] : i64
// CHECK: %[[rhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[rhsZero]] : f64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[lhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[lhsZero]] : f64
// CHECK-DAG: %[[rhsZero:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Boolean array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x?x!bmodelica.bool>, %[[arg1:.*]]: !bmodelica.array<3x?x!bmodelica.bool>) -> !bmodelica.array<3x?x!bmodelica.bool>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim1:.*]] = bmodelica.dim %[[arg1]], %[[c1]]
// CHECK-DAG:   %[[dim1_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim1]]
// CHECK-DAG:   cf.assert %[[dim1_cmp]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.alloc %[[arg0_dim1]] : <3x?x!bmodelica.bool>
// CHECK-DAG:   %[[result_dim0:.*]] = bmodelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = bmodelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[lhs:.*]] = bmodelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = bmodelica.load %[[arg1]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !bmodelica.bool to i1
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !bmodelica.bool to i1
// CHECK-DAG:           %[[lhs_ne:.*]] = arith.cmpi ne, %[[lhs_casted]], %[[false]] : i1
// CHECK-DAG:           %[[rhs_ne:.*]] = arith.cmpi ne, %[[rhs_casted]], %[[false]] : i1
// CHECK:               %[[and:.*]] = arith.andi %[[lhs_ne]], %[[rhs_ne]] : i1
// CHECK:               %[[and_casted:.*]] = builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK:               bmodelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[and_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<3x?x!bmodelica.bool>, %arg1 : !bmodelica.array<3x?x!bmodelica.bool>) -> !bmodelica.array<3x?x!bmodelica.bool> {
    %0 = bmodelica.and %arg0, %arg1 : (!bmodelica.array<3x?x!bmodelica.bool>, !bmodelica.array<3x?x!bmodelica.bool>) -> !bmodelica.array<3x?x!bmodelica.bool>
    func.return %0 : !bmodelica.array<3x?x!bmodelica.bool>
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[arg0]], %[[zero]] : index
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[arg1]], %[[zero]] : index
// CHECK: %[[result:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : index, %arg1 : index) -> i1 {
    %0 = bmodelica.and %arg0, %arg1 : (index, index) -> i1
    func.return %0 : i1
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[arg0]], %[[zero]] : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[arg1]], %[[zero]] : i64
// CHECK: %[[result:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : i64, %arg1 : i64) -> i1 {
    %0 = bmodelica.and %arg0, %arg1 : (i64, i64) -> i1
    func.return %0 : i1
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[lhs:.*]] = arith.cmpf one, %[[arg0]], %[[zero]] : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[arg1]], %[[zero]] : f64
// CHECK: %[[result:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : f64, %arg1 : f64) -> i1 {
    %0 = bmodelica.and %arg0, %arg1 : (f64, f64) -> i1
    func.return %0 : i1
}
