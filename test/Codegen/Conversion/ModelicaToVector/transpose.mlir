// RUN: modelica-opt %s --split-input-file --convert-modelica-to-vector --cse | FileCheck %s

// Integer matrix operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x5x!modelica.int>) -> !modelica.array<5x3x!modelica.int>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = vector.transpose %[[arg0_vector]], [1, 0] : vector<3x5xi64> to vector<5x3xi64>
// CHECK:       %[[result_array:.*]] = modelica.alloc : !modelica.array<5x3x!modelica.int>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x5x!modelica.int>) -> !modelica.array<5x3x!modelica.int> {
    %0 = modelica.transpose %arg0 : !modelica.array<3x5x!modelica.int> -> !modelica.array<5x3x!modelica.int>
    func.return %0 : !modelica.array<5x3x!modelica.int>
}

// -----

// Real matrix operand

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x5x!modelica.real>) -> !modelica.array<5x3x!modelica.real>
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x5x!modelica.real> to memref<3x5xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]], %[[c0_index]]]
// CHECK:       %[[result_vector:.*]] = vector.transpose %[[arg0_vector]], [1, 0] : vector<3x5xf64> to vector<5x3xf64>
// CHECK:       %[[result_array:.*]] = modelica.alloc : !modelica.array<5x3x!modelica.real>
// CHECK:       %[[result_memref:.*]] = builtin.unrealized_conversion_cast %[[result_array]] : !modelica.array<5x3x!modelica.real> to memref<5x3xf64>
// CHECK:       vector.transfer_write %[[result_vector]], %[[result_memref]][%[[c0_index]], %[[c0_index]]]
// CHECK:       return %[[result_array]]

func.func @foo(%arg0 : !modelica.array<3x5x!modelica.real>) -> !modelica.array<5x3x!modelica.real> {
    %0 = modelica.transpose %arg0 : !modelica.array<3x5x!modelica.real> -> !modelica.array<5x3x!modelica.real>
    func.return %0 : !modelica.array<5x3x!modelica.real>
}
