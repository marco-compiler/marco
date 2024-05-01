// RUN: modelica-opt %s --split-input-file --convert-modelica-to-vector --cse | FileCheck %s

// Integer 1-D array operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x!bmodelica.int>) -> !bmodelica.int
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<3x!bmodelica.int> to memref<3xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]]
// CHECK-DAG:   %[[acc:.*]] = arith.constant 1 : i64
// CHECK:       %[[result:.*]] = vector.reduction <mul>, %[[arg0_vector]], %[[acc]] : vector<3xi64> into i64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK:       return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.array<3x!bmodelica.int>) -> !bmodelica.int {
    %0 = bmodelica.product %arg0 : !bmodelica.array<3x!bmodelica.int> -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// Real 1-D array operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x!bmodelica.real>) -> !bmodelica.real
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<3x!bmodelica.real> to memref<3xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]]
// CHECK-DAG:   %[[acc:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:       %[[result:.*]] = vector.reduction <mul>, %[[arg0_vector]], %[[acc]] : vector<3xf64> into f64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK:       return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.array<3x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.product %arg0 : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.real
    func.return %0 : !bmodelica.real
}
