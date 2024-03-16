// RUN: modelica-opt %s --split-input-file --convert-modelica-to-vector --cse | FileCheck %s

// Integer array operands.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x!modelica.int>, %[[arg1:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<3x!modelica.int>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x!modelica.int> to memref<3xi64>
// CHECK-DAG:   %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.array<3x!modelica.int> to memref<3xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG:   %[[lhs_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]], %[[c0_i64]]
// CHECK-DAG:   %[[rhs_vector:.*]] = vector.transfer_read %[[arg1_casted]][%[[c0_index]]], %[[c0_i64]]
// CHECK:       %[[result_vector:.*]] = arith.addi %[[lhs_vector]], %[[rhs_vector]] : vector<3xi64>
// CHECK:       %[[result_array:.*]] = modelica.alloc  : <3x!modelica.int>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<3x!modelica.int> to memref<3xi64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x!modelica.int>, %arg1 : !modelica.array<3x!modelica.int>) -> !modelica.array<3x!modelica.int> {
    %0 = modelica.add %arg0, %arg1 : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.array<3x!modelica.int>
    func.return %0 : !modelica.array<3x!modelica.int>
}

// -----

// Real array operands.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x!modelica.real>, %[[arg1:.*]]: !modelica.array<3x!modelica.real>) -> !modelica.array<3x!modelica.real>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x!modelica.real> to memref<3xf64>
// CHECK-DAG:   %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.array<3x!modelica.real> to memref<3xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c0_f64:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:   %[[lhs_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]], %[[c0_f64]]
// CHECK-DAG:   %[[rhs_vector:.*]] = vector.transfer_read %[[arg1_casted]][%[[c0_index]]], %[[c0_f64]]
// CHECK:       %[[result_vector:.*]] = arith.addf %[[lhs_vector]], %[[rhs_vector]] : vector<3xf64>
// CHECK:       %[[result_array:.*]] = modelica.alloc  : <3x!modelica.real>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<3x!modelica.real> to memref<3xf64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x!modelica.real>, %arg1 : !modelica.array<3x!modelica.real>) -> !modelica.array<3x!modelica.real> {
    %0 = modelica.add %arg0, %arg1 : (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.real>) -> !modelica.array<3x!modelica.real>
    func.return %0 : !modelica.array<3x!modelica.real>
}
