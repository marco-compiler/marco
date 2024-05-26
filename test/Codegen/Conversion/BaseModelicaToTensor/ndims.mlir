// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// CHECK-LABEL: @staticArray
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x3xi64>) -> i64
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = bmodelica.cast %[[result]] : index -> i64
// CHECK: return %[[result_casted]]

func.func @staticArray(%arg0: tensor<5x3xi64>) -> i64 {
    %0 = bmodelica.ndims %arg0 : tensor<5x3xi64> -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?xi64>) -> i64
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = bmodelica.cast %[[result]] : index -> i64
// CHECK: return %[[result_casted]]

func.func @dynamicArray(%arg0: tensor<?x?xi64>) -> i64 {
    %0 = bmodelica.ndims %arg0 : tensor<?x?xi64> -> i64
    func.return %0 : i64
}
