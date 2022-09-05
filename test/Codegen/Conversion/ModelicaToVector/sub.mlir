// RUN: modelica-opt %s --split-input-file --convert-modelica-to-vector --cse | FileCheck %s

// Integer array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x5x!modelica.int>, %[[arg1:.*]]: !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.int>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-DAG:   %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[lhs_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK-DAG:   %[[rhs_vector:.*]] = vector.transfer_read %[[arg1_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = arith.subi %[[lhs_vector]], %[[rhs_vector]] : vector<3x5xi64>
// CHECK:       %[[result_array:.*]] = modelica.alloc  : !modelica.array<3x5x!modelica.int>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x5x!modelica.int>, %arg1 : !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.int> {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.array<3x5x!modelica.int>, !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.int>
    func.return %0 : !modelica.array<3x5x!modelica.int>
}

// -----

// Real array operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x5x!modelica.real>, %[[arg1:.*]]: !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x5x!modelica.real> to memref<3x5xf64>
// CHECK-DAG:   %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.array<3x5x!modelica.real> to memref<3x5xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[lhs_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK-DAG:   %[[rhs_vector:.*]] = vector.transfer_read %[[arg1_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = arith.subf %[[lhs_vector]], %[[rhs_vector]] : vector<3x5xf64>
// CHECK:       %[[result_array:.*]] = modelica.alloc  : !modelica.array<3x5x!modelica.real>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<3x5x!modelica.real> to memref<3x5xf64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x5x!modelica.real>, %arg1 : !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real> {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.array<3x5x!modelica.real>, !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>
    func.return %0 : !modelica.array<3x5x!modelica.real>
}
