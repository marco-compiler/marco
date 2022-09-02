// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Booleans operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.bool, %[[arg1:.*]]: !modelica.bool) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.bool to i1
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.bool to i1
// CHECK-DAG: %[[false:.*]] = arith.constant false
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[false]] : i1
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[false]] : i1
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.bool, %arg1 : !modelica.bool) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Integers operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[zero]] : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[zero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Reals operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[zero]] : f64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[zero]] : f64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[lhsZero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[lhsZero]] : i64
// CHECK: %[[rhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[rhsZero]] : f64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK-DAG: %[[lhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[lhsZero]] : f64
// CHECK-DAG: %[[rhsZero:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Boolean array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x?x!modelica.bool>, %[[arg1:.*]]: !modelica.array<3x?x!modelica.bool>) -> !modelica.array<3x?x!modelica.bool>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[arg1_dim1:.*]] = modelica.dim %[[arg1]], %[[c1]]
// CHECK-DAG:   %[[dim1_cmp:.*]] = arith.cmpi eq, %[[arg0_dim1]], %[[arg1_dim1]]
// CHECK-DAG:   cf.assert %[[dim1_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.alloc %[[arg0_dim1]] : !modelica.array<3x?x!modelica.bool>
// CHECK-DAG:   %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:   %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] {
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[lhs:.*]] = modelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.load %[[arg1]][%[[index_0]], %[[index_1]]]
// CHECK-DAG:           %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.bool to i1
// CHECK-DAG:           %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.bool to i1
// CHECK-DAG:           %[[lhs_ne:.*]] = arith.cmpi ne, %[[lhs_casted]], %[[false]] : i1
// CHECK-DAG:           %[[rhs_ne:.*]] = arith.cmpi ne, %[[rhs_casted]], %[[false]] : i1
// CHECK:               %[[and:.*]] = arith.andi %[[lhs_ne]], %[[rhs_ne]] : i1
// CHECK:               %[[and_casted:.*]] = builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK:               modelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[and_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<3x?x!modelica.bool>, %arg1 : !modelica.array<3x?x!modelica.bool>) -> !modelica.array<3x?x!modelica.bool> {
    %0 = modelica.and %arg0, %arg1 : (!modelica.array<3x?x!modelica.bool>, !modelica.array<3x?x!modelica.bool>) -> !modelica.array<3x?x!modelica.bool>
    func.return %0 : !modelica.array<3x?x!modelica.bool>
}