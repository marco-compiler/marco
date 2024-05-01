// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// Scalar indices.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<6x5x4x3x2x!bmodelica.int>) -> !bmodelica.array<4x3x2x!bmodelica.int>
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<6x5x4x3x2x!bmodelica.int> to memref<6x5x4x3x2xi64>
// CHECK: %[[c3:.*]] = arith.constant 3 : index
// CHECK: %[[c2:.*]] = arith.constant 2 : index
// CHECK: %[[subview:.*]] = memref.subview %[[memref]][%[[c3]], %[[c2]], 0, 0, 0] [1, 1, 4, 3, 2] [1, 1, 1, 1, 1] : memref<6x5x4x3x2xi64> to memref<4x3x2xi64, strided<[6, 2, 1], offset: ?>>
// CHECK: %[[result:.*]] = memref.cast %[[subview]] : memref<4x3x2xi64, strided<[6, 2, 1], offset: ?>> to memref<4x3x2xi64>
// CHECK: %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<4x3x2xi64> to !bmodelica.array<4x3x2x!bmodelica.int>
// CHECK: return %[[result_cast]] : !bmodelica.array<4x3x2x!bmodelica.int>

func.func @foo(%arg0: !bmodelica.array<6x5x4x3x2x!bmodelica.int>) -> !bmodelica.array<4x3x2x!bmodelica.int> {
    %0 = arith.constant 3 : index
    %1 = arith.constant 2 : index
    %2 = bmodelica.subscription %arg0[%0, %1] : !bmodelica.array<6x5x4x3x2x!bmodelica.int>, index, index -> !bmodelica.array<4x3x2x!bmodelica.int>
    func.return %2 : !bmodelica.array<4x3x2x!bmodelica.int>
}

// -----

// Slicing.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<6x5x4x3x2x!bmodelica.int>) -> !bmodelica.array<?x?x4x3x2x!bmodelica.int>
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<6x5x4x3x2x!bmodelica.int> to memref<6x5x4x3x2xi64>
// CHECK-DAG: %[[int_range:.*]] = bmodelica.constant #bmodelica.int_range<3, 10, 2>
// CHECK-DAG: %[[int_range_begin:.*]] = bmodelica.range_begin %[[int_range]]
// CHECK-DAG: %[[int_range_begin_casted:.*]] = bmodelica.cast %[[int_range_begin]] : !bmodelica.int -> index
// CHECK-DAG: %[[int_range_step:.*]] = bmodelica.range_step %[[int_range]]
// CHECK-DAG: %[[int_range_step_casted:.*]] = bmodelica.cast %[[int_range_step]] : !bmodelica.int -> index
// CHECK-DAG: %[[int_range_size:.*]] = bmodelica.range_size %[[int_range]]
// CHECK-DAG: %[[real_range:.*]] = bmodelica.constant #bmodelica.real_range<3.000000e+00, 1.000000e+01, 2.000000e+00>
// CHECK-DAG: %[[real_range_begin:.*]] = bmodelica.range_begin %[[real_range]]
// CHECK-DAG: %[[real_range_begin_casted:.*]] = bmodelica.cast %[[real_range_begin]] : !bmodelica.real -> index
// CHECK-DAG: %[[real_range_step:.*]] = bmodelica.range_step %[[real_range]]
// CHECK-DAG: %[[real_range_step_casted:.*]] = bmodelica.cast %[[real_range_step]] : !bmodelica.real -> index
// CHECK-DAG: %[[real_range_size:.*]] = bmodelica.range_size %[[real_range]]
// CHECK: %[[subview:.*]] = memref.subview %[[memref]][%[[int_range_begin_casted]], %[[real_range_begin_casted]], 0, 0, 0] [%[[int_range_size]], %[[real_range_size]], 4, 3, 2] [%[[int_range_step_casted]], %[[real_range_step_casted]], 1, 1, 1] : memref<6x5x4x3x2xi64> to memref<?x?x4x3x2xi64, strided<[?, ?, 6, 2, 1], offset: ?>>
// CHECK: %[[result:.*]] = memref.cast %[[subview]] : memref<?x?x4x3x2xi64, strided<[?, ?, 6, 2, 1], offset: ?>> to memref<?x?x4x3x2xi64>
// CHECK: %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<?x?x4x3x2xi64> to !bmodelica.array<?x?x4x3x2x!bmodelica.int>
// CHECK: return %[[result_cast]] : !bmodelica.array<?x?x4x3x2x!bmodelica.int>

func.func @foo(%arg0: !bmodelica.array<6x5x4x3x2x!bmodelica.int>) -> !bmodelica.array<?x?x4x3x2x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.int_range<3, 10, 2> : !bmodelica<range !bmodelica.int>
    %1 = bmodelica.constant #bmodelica.real_range<3.0, 10.0, 2.0> : !bmodelica<range !bmodelica.real>
    %2 = bmodelica.subscription %arg0[%0, %1] : !bmodelica.array<6x5x4x3x2x!bmodelica.int>, !bmodelica<range !bmodelica.int>, !bmodelica<range !bmodelica.real> -> !bmodelica.array<?x?x4x3x2x!bmodelica.int>
    func.return %2 : !bmodelica.array<?x?x4x3x2x!bmodelica.int>
}
