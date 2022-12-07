// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @_Mmod_i64_i64_i64 : (i64, i64) -> i64

// CHECK-LABEL: @integers
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[result:.*]] = modelica.call @_Mmod_i64_i64_i64(%[[arg0_casted]], %[[arg1_casted]]) : (i64, i64) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @integers(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mod %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// CHECK: modelica.runtime_function @_Mmod_f64_f64_f64 : (f64, f64) -> f64

// CHECK-LABEL: @reals
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @_Mmod_f64_f64_f64(%[[arg0_casted]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @reals(%arg0: !modelica.real, %arg1: !modelica.real) -> !modelica.real {
    %0 = modelica.mod %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// CHECK: modelica.runtime_function @_Mmod_f64_f64_f64 : (f64, f64) -> f64

// CHECK-LABEL: @integerReal
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted_1:.*]] = modelica.cast %[[arg0]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[arg0_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg0_casted_1]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @_Mmod_f64_f64_f64(%[[arg0_casted_2]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @integerReal(%arg0: !modelica.int, %arg1: !modelica.real) -> !modelica.real {
    %0 = modelica.mod %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}

// -----

// CHECK: modelica.runtime_function @_Mmod_f64_f64_f64 : (f64, f64) -> f64

// CHECK-LABEL: @realInteger
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted_1:.*]] = modelica.cast %[[arg1]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[arg1_casted_2:.*]] = builtin.unrealized_conversion_cast %[[arg1_casted_1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @_Mmod_f64_f64_f64(%[[arg0_casted]], %[[arg1_casted_2]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @realInteger(%arg0: !modelica.real, %arg1: !modelica.int) -> !modelica.real {
    %0 = modelica.mod %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.real
    func.return %0 : !modelica.real
}
