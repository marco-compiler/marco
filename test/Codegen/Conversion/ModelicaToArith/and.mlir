// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @boolean
// CHECK-SAME: (%[[arg0:.*]]: !modelica.bool, %[[arg1:.*]]: !modelica.bool) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.bool to i1
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.bool to i1
// CHECK: %[[lhsZero:.*]] = arith.constant false
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[lhsZero]] : i1
// CHECK: %[[rhsZero:.*]] = arith.constant false
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i1
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @booleans(%arg0 : !modelica.bool, %arg1 : !modelica.bool) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @integers
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[lhsZero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[lhsZero]] : i64
// CHECK: %[[rhsZero:.*]] = arith.constant 0 : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @integers(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @reals
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[lhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[lhsZero]] : f64
// CHECK: %[[rhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[rhsZero]] : f64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @reals(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @integerReal
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

func.func @integerReal(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @realInteger
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[lhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[lhsZero]] : f64
// CHECK: %[[rhsZero:.*]] = arith.constant 0 : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i64
// CHECK: %[[and:.*]] = arith.andi %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @realInteger(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.and %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}
