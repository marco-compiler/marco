// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @staticArray
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.array<2x!modelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK:       %[[result:.*]] = memref.alloc() : memref<2xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<2xi64> to !modelica.array<2x!modelica.int>
// CHECK-DAG:   %[[lower_bound:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[upper_bound:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[index:.*]] = %[[lower_bound]] to %[[upper_bound]] step %[[step]] {
// CHECK-NEXT:      %[[dimension:.*]] = memref.dim %[[arg0_casted]], %[[index]]
// CHECK-NEXT:      %[[dimension_cast_1:.*]] = modelica.cast %[[dimension]] : index -> !modelica.int
// CHECK-NEXT:      %[[dimension_cast_2:.*]] = builtin.unrealized_conversion_cast %[[dimension_cast_1]] : !modelica.int to i64
// CHECK-NEXT:      memref.store %[[dimension_cast_2]], %[[result]][%[[index]]]
// CHECK-NEXT:  }
// CHECK:       return %[[result_casted]]

func.func @staticArray(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.array<2x!modelica.int> {
    %0 = modelica.size %arg0 : !modelica.array<5x3x!modelica.int> -> !modelica.array<2x!modelica.int>
    func.return %0 : !modelica.array<2x!modelica.int>
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x?x!modelica.int>) -> !modelica.array<2x!modelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<?x?x!modelica.int> to memref<?x?xi64>
// CHECK:       %[[result:.*]] = memref.alloc() : memref<2xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<2xi64> to !modelica.array<2x!modelica.int>
// CHECK-DAG:   %[[lower_bound:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[upper_bound:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[index:.*]] = %[[lower_bound]] to %[[upper_bound]] step %[[step]] {
// CHECK-NEXT:      %[[dimension:.*]] = memref.dim %[[arg0_casted]], %[[index]]
// CHECK-NEXT:      %[[dimension_cast_1:.*]] = modelica.cast %[[dimension]] : index -> !modelica.int
// CHECK-NEXT:      %[[dimension_cast_2:.*]] = builtin.unrealized_conversion_cast %[[dimension_cast_1]] : !modelica.int to i64
// CHECK-NEXT:      memref.store %[[dimension_cast_2]], %[[result]][%[[index]]]
// CHECK-NEXT:  }
// CHECK:       return %[[result_casted]]

func.func @dynamicArray(%arg0: !modelica.array<?x?x!modelica.int>) -> !modelica.array<2x!modelica.int> {
    %0 = modelica.size %arg0 : !modelica.array<?x?x!modelica.int> -> !modelica.array<2x!modelica.int>
    func.return %0 : !modelica.array<2x!modelica.int>
}

// -----

// CHECK-LABEL: @staticArrayDimension
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>, %[[arg1:.*]]: index) -> index
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK: %[[result:.*]] = memref.dim %[[arg0_casted]], %[[arg1]]
// CHECK: return %[[result]]

func.func @staticArrayDimension(%arg0: !modelica.array<5x3x!modelica.int>, %arg1: index) -> index {
    %0 = modelica.size %arg0, %arg1 : (!modelica.array<5x3x!modelica.int>, index) -> index
    func.return %0 : index
}

// -----

// CHECK-LABEL: @dynamicArrayDimension
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x?x!modelica.int>, %[[arg1:.*]]: index) -> index
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<?x?x!modelica.int> to memref<?x?xi64>
// CHECK: %[[result:.*]] = memref.dim %[[arg0_casted]], %[[arg1]]
// CHECK: return %[[result]]

func.func @dynamicArrayDimension(%arg0: !modelica.array<?x?x!modelica.int>, %arg1: index) -> index {
    %0 = modelica.size %arg0, %arg1 : (!modelica.array<?x?x!modelica.int>, index) -> index
    func.return %0 : index
}
