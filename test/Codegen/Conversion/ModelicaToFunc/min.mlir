// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: func.func private @_MminArray_i64_ai64(memref<*xi64>) -> i64
// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK: %[[arg0_unranked:.*]] = memref.cast %[[arg0_casted]] : memref<5x3xi64> to memref<*xi64>
// CHECK: %[[result:.*]] = call @_MminArray_i64_ai64(%[[arg0_unranked]]) : (memref<*xi64>) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.int {
    %0 = bmodelica.min %arg0 : !bmodelica.array<5x3x!bmodelica.int> -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK: func.func private @_MminScalars_i64_i64_i64(i64, i64) -> i64
// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[result:.*]] = call @_MminScalars_i64_i64_i64(%[[arg0_casted]], %[[arg1_casted]]) : (i64, i64) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.min %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    func.return %0 : !bmodelica.int
}
