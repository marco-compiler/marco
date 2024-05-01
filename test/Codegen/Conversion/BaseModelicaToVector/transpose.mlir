// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-vector --cse | FileCheck %s

// Integer matrix operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x5x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<3x5x!bmodelica.int> to memref<3x5xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = vector.transpose %[[arg0_vector]], [1, 0] : vector<3x5xi64> to vector<5x3xi64>
// CHECK:       %[[result_array:.*]] = bmodelica.alloc : <5x3x!bmodelica.int>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !bmodelica.array<3x5x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int> {
    %0 = bmodelica.transpose %arg0 : !bmodelica.array<3x5x!bmodelica.int> -> !bmodelica.array<5x3x!bmodelica.int>
    func.return %0 : !bmodelica.array<5x3x!bmodelica.int>
}

// -----

// Real matrix operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x5x!bmodelica.real>) -> !bmodelica.array<5x3x!bmodelica.real>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<3x5x!bmodelica.real> to memref<3x5xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = vector.transpose %[[arg0_vector]], [1, 0] : vector<3x5xf64> to vector<5x3xf64>
// CHECK:       %[[result_array:.*]] = bmodelica.alloc : <5x3x!bmodelica.real>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !bmodelica.array<5x3x!bmodelica.real> to memref<5x3xf64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !bmodelica.array<3x5x!bmodelica.real>) -> !bmodelica.array<5x3x!bmodelica.real> {
    %0 = bmodelica.transpose %arg0 : !bmodelica.array<3x5x!bmodelica.real> -> !bmodelica.array<5x3x!bmodelica.real>
    func.return %0 : !bmodelica.array<5x3x!bmodelica.real>
}
