// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[result:.*]] = arith.divsi %[[arg0_casted]], %[[arg1_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @Integer(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @Real
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_casted]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @Real(%arg0 : !bmodelica.real, %arg1 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @IntegerReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[arg0_f64:.*]] = arith.sitofp %[[arg0_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_f64]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @IntegerReal(%arg0 : !bmodelica.int, %arg1 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @RealInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[arg1_f64:.*]] = arith.sitofp %[[arg1_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_casted]], %[[arg1_f64]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @RealInteger(%arg0 : !bmodelica.real, %arg1 : !bmodelica.int) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @IntegerOperandsAndRealResult
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[lhs:.*]] = arith.sitofp %[[arg0_casted]] : i64 to f64
// CHECK-DAG: %[[rhs:.*]] = arith.sitofp %[[arg1_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.divf %[[lhs]], %[[rhs]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @IntegerOperandsAndRealResult(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mlirIndex
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> index
// CHECK: %[[result:.*]] = arith.divsi %[[arg0]], %[[arg1]] : index
// CHECK: return %[[result]] : index

func.func @mlirIndex(%arg0 : index, %arg1 : index) -> index {
    %0 = bmodelica.div %arg0, %arg1 : (index, index) -> index
    func.return %0 : index
}

// -----

// CHECK-LABEL: @mlirInteger
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = arith.divsi %[[arg0]], %[[arg1]] : i64
// CHECK: return %[[result]] : i64

func.func @mlirInteger(%arg0 : i64, %arg1 : i64) -> i64 {
    %0 = bmodelica.div %arg0, %arg1 : (i64, i64) -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @mlirFloat
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0]], %[[arg1]] : f64
// CHECK: return %[[result]] : f64

func.func @mlirFloat(%arg0 : f64, %arg1 : f64) -> f64 {
    %0 = bmodelica.div %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}
