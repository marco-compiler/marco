// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @Integer(%arg0 : !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.neg %arg0 : !bmodelica.int -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @Real
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = arith.negf %[[arg0_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @Real(%arg0 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.neg %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mlirIndex
// CHECK-SAME: (%[[arg0:.*]]: index) -> index
// CHECK: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0]] : index
// CHECK: return %[[result]]

func.func @mlirIndex(%arg0 : index) -> index {
    %0 = bmodelica.neg %arg0 : index -> index
    func.return %0 : index
}

// -----

// CHECK-LABEL: @mlirInteger
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.subi %[[zero]], %[[arg0]] : i64
// CHECK: return %[[result]]

func.func @mlirInteger(%arg0 : i64) -> i64 {
    %0 = bmodelica.neg %arg0 : i64 -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @mlirFloat
// CHECK-SAME: (%[[arg0:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = arith.negf %[[arg0]] : f64
// CHECK: return %[[result]]

func.func @mlirFloat(%arg0 : f64) -> f64 {
    %0 = bmodelica.neg %arg0 : f64 -> f64
    func.return %0 : f64
}
