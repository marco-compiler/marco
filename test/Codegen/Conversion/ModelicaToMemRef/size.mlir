// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @staticArray
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<2x!bmodelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK:       %[[result:.*]] = memref.alloc() : memref<2xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK-DAG:   %[[lower_bound:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[upper_bound:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[index:.*]] = %[[lower_bound]] to %[[upper_bound]] step %[[step]] {
// CHECK-NEXT:      %[[dimension:.*]] = memref.dim %[[arg0_casted]], %[[index]]
// CHECK-NEXT:      %[[dimension_cast_1:.*]] = bmodelica.cast %[[dimension]] : index -> !bmodelica.int
// CHECK-NEXT:      %[[dimension_cast_2:.*]] = builtin.unrealized_conversion_cast %[[dimension_cast_1]] : !bmodelica.int to i64
// CHECK-NEXT:      memref.store %[[dimension_cast_2]], %[[result]][%[[index]]]
// CHECK-NEXT:  }
// CHECK:       return %[[result_casted]]

func.func @staticArray(%arg0: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<2x!bmodelica.int> {
    %0 = bmodelica.size %arg0 : !bmodelica.array<5x3x!bmodelica.int> -> !bmodelica.array<2x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x!bmodelica.int>
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<?x?x!bmodelica.int>) -> !bmodelica.array<2x!bmodelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<?x?x!bmodelica.int> to memref<?x?xi64>
// CHECK:       %[[result:.*]] = memref.alloc() : memref<2xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK-DAG:   %[[lower_bound:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[upper_bound:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[index:.*]] = %[[lower_bound]] to %[[upper_bound]] step %[[step]] {
// CHECK-NEXT:      %[[dimension:.*]] = memref.dim %[[arg0_casted]], %[[index]]
// CHECK-NEXT:      %[[dimension_cast_1:.*]] = bmodelica.cast %[[dimension]] : index -> !bmodelica.int
// CHECK-NEXT:      %[[dimension_cast_2:.*]] = builtin.unrealized_conversion_cast %[[dimension_cast_1]] : !bmodelica.int to i64
// CHECK-NEXT:      memref.store %[[dimension_cast_2]], %[[result]][%[[index]]]
// CHECK-NEXT:  }
// CHECK:       return %[[result_casted]]

func.func @dynamicArray(%arg0: !bmodelica.array<?x?x!bmodelica.int>) -> !bmodelica.array<2x!bmodelica.int> {
    %0 = bmodelica.size %arg0 : !bmodelica.array<?x?x!bmodelica.int> -> !bmodelica.array<2x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x!bmodelica.int>
}

// -----

// CHECK-LABEL: @staticArrayDimension
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>, %[[arg1:.*]]: index) -> index
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK: %[[result:.*]] = memref.dim %[[arg0_casted]], %[[arg1]]
// CHECK: return %[[result]]

func.func @staticArrayDimension(%arg0: !bmodelica.array<5x3x!bmodelica.int>, %arg1: index) -> index {
    %0 = bmodelica.size %arg0, %arg1 : (!bmodelica.array<5x3x!bmodelica.int>, index) -> index
    func.return %0 : index
}

// -----

// CHECK-LABEL: @dynamicArrayDimension
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<?x?x!bmodelica.int>, %[[arg1:.*]]: index) -> index
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<?x?x!bmodelica.int> to memref<?x?xi64>
// CHECK: %[[result:.*]] = memref.dim %[[arg0_casted]], %[[arg1]]
// CHECK: return %[[result]]

func.func @dynamicArrayDimension(%arg0: !bmodelica.array<?x?x!bmodelica.int>, %arg1: index) -> index {
    %0 = bmodelica.size %arg0, %arg1 : (!bmodelica.array<?x?x!bmodelica.int>, index) -> index
    func.return %0 : index
}
