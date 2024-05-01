// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @staticArray
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.int
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = bmodelica.cast %[[result]] : index -> !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @staticArray(%arg0: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.int {
    %0 = bmodelica.ndims %arg0 : !bmodelica.array<5x3x!bmodelica.int> -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<?x?x!bmodelica.int>) -> !bmodelica.int
// CHECK: %[[result:.*]] = arith.constant 2 : index
// CHECK: %[[result_casted:.*]] = bmodelica.cast %[[result]] : index -> !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @dynamicArray(%arg0: !bmodelica.array<?x?x!bmodelica.int>) -> !bmodelica.int {
    %0 = bmodelica.ndims %arg0 : !bmodelica.array<?x?x!bmodelica.int> -> !bmodelica.int
    func.return %0 : !bmodelica.int
}
