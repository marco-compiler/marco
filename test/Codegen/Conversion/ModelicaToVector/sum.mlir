// RUN: modelica-opt %s --split-input-file --convert-modelica-to-vector --cse | FileCheck %s

// Integer 1-D array operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.int
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x!modelica.int> to memref<3xi64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]]
// CHECK-DAG:   %[[acc:.*]] = arith.constant 0 : i64
// CHECK:       %[[result:.*]] = vector.reduction <add>, %[[arg0_vector]], %[[acc]] : vector<3xi64> into i64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK:       return %[[result_casted]]

func.func @foo(%arg0 : !modelica.array<3x!modelica.int>) -> !modelica.int {
    %0 = modelica.sum %arg0 : !modelica.array<3x!modelica.int> -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// Real 1-D array operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x!modelica.real>) -> !modelica.real
// CHECK-DAG:   %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x!modelica.real> to memref<3xf64>
// CHECK-DAG:   %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[arg0_vector:.*]] = vector.transfer_read %[[arg0_casted]][%[[c0_index]]]
// CHECK-DAG:   %[[acc:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[result:.*]] = vector.reduction <add>, %[[arg0_vector]], %[[acc]] : vector<3xf64> into f64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK:       return %[[result_casted]]

func.func @foo(%arg0 : !modelica.array<3x!modelica.real>) -> !modelica.real {
    %0 = modelica.sum %arg0 : !modelica.array<3x!modelica.real> -> !modelica.real
    func.return %0 : !modelica.real
}
