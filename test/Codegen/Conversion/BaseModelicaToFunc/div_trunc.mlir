// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK: func.func private @_Mdiv_i64_i64_i64(i64, i64) -> i64

// CHECK-LABEL: @integers
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[result:.*]] = call @_Mdiv_i64_i64_i64(%[[arg0_casted]], %[[arg1_casted]]) : (i64, i64) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @integers(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div_trunc %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK: func.func private @_Mdiv_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @reals
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Mdiv_f64_f64_f64(%[[arg0_casted]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @reals(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div_trunc %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK: func.func private @_Mdiv_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @integerReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted_1:.*]] = bmodelica.cast %[[arg0]] : !bmodelica.int -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg0_casted_1]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Mdiv_f64_f64_f64(%[[arg0_casted_2]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @integerReal(%arg0: !bmodelica.int, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div_trunc %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK: func.func private @_Mdiv_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @realInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted_1:.*]] = bmodelica.cast %[[arg1]] : !bmodelica.int -> !bmodelica.real
// CHECK-DAG: %[[arg1_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg1_casted_1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Mdiv_f64_f64_f64(%[[arg0_casted]], %[[arg1_casted_2]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @realInteger(%arg0: !bmodelica.real, %arg1: !bmodelica.int) -> !bmodelica.real {
    %0 = bmodelica.div_trunc %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}
