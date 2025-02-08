// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// CHECK-LABEL: @Boolean
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool, %[[arg1:.*]]: !bmodelica.bool) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.bool to i1
// CHECK-DAG: %[[false:.*]] = arith.constant false
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[false]] : i1
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[false]] : i1
// CHECK: %[[and:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @Boolean(%arg0 : !bmodelica.bool, %arg1 : !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.or %arg0, %arg1 : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @Integer
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[zero]] : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[zero]] : i64
// CHECK: %[[and:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @Integer(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.or %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @Real
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[zero]] : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[zero]] : f64
// CHECK: %[[and:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @Real(%arg0 : !bmodelica.real, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.or %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @IntegerReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK-DAG: %[[lhsZero:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpi ne, %[[x]], %[[lhsZero]] : i64
// CHECK-DAG: %[[rhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpf one, %[[y]], %[[rhsZero]] : f64
// CHECK: %[[and:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @IntegerReal(%arg0 : !bmodelica.int, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.or %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @RealInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[lhsZero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[lhs:.*]] = arith.cmpf one, %[[x]], %[[lhsZero]] : f64
// CHECK-DAG: %[[rhsZero:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[rhs:.*]] = arith.cmpi ne, %[[y]], %[[rhsZero]] : i64
// CHECK: %[[and:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[and]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @RealInteger(%arg0 : !bmodelica.real, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.or %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @mlirIndex
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[arg0]], %[[zero]] : index
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[arg1]], %[[zero]] : index
// CHECK: %[[result:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @mlirIndex(%arg0 : index, %arg1 : index) -> i1 {
    %0 = bmodelica.or %arg0, %arg1 : (index, index) -> i1
    func.return %0 : i1
}

// -----

// CHECK-LABEL: @mlirInteger
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[lhs:.*]] = arith.cmpi ne, %[[arg0]], %[[zero]] : i64
// CHECK: %[[rhs:.*]] = arith.cmpi ne, %[[arg1]], %[[zero]] : i64
// CHECK: %[[result:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @mlirInteger(%arg0 : i64, %arg1 : i64) -> i1 {
    %0 = bmodelica.or %arg0, %arg1 : (i64, i64) -> i1
    func.return %0 : i1
}

// -----

// CHECK-LABEL: @mlirFloat
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[lhs:.*]] = arith.cmpf one, %[[arg0]], %[[zero]] : f64
// CHECK: %[[rhs:.*]] = arith.cmpf one, %[[arg1]], %[[zero]] : f64
// CHECK: %[[result:.*]] = arith.ori %[[lhs]], %[[rhs]] : i1
// CHECK: return %[[result]] : i1

func.func @mlirFloat(%arg0 : f64, %arg1 : f64) -> i1 {
    %0 = bmodelica.or %arg0, %arg1 : (f64, f64) -> i1
    func.return %0 : i1
}
