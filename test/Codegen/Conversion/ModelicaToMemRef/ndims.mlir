// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @staticArray
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.int
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = modelica.cast %[[result]] : index -> !modelica.int
// CHECK: return %[[result_casted]]

func.func @staticArray(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.int {
    %0 = modelica.ndims %arg0 : !modelica.array<5x3x!modelica.int> -> !modelica.int
    func.return %0 : !modelica.int
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x?x!modelica.int>) -> !modelica.int
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = modelica.cast %[[result]] : index -> !modelica.int
// CHECK: return %[[result_casted]]

func.func @dynamicArray(%arg0: !modelica.array<?x?x!modelica.int>) -> !modelica.int {
    %0 = modelica.ndims %arg0 : !modelica.array<?x?x!modelica.int> -> !modelica.int
    func.return %0 : !modelica.int
}
