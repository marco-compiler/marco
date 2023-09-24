// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// Scalar indices.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<4x3x2x!modelica.int>
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<6x5x4x3x2x!modelica.int> to memref<6x5x4x3x2xi64>
// CHECK: %[[c3:.*]] = arith.constant 3 : index
// CHECK: %[[c2:.*]] = arith.constant 2 : index
// CHECK: %[[subview:.*]] = memref.subview %[[memref]][%[[c3]], %[[c2]], 0, 0, 0] [1, 1, 4, 3, 2] [1, 1, 1, 1, 1] : memref<6x5x4x3x2xi64> to memref<4x3x2xi64, strided<[6, 2, 1], offset: ?>>
// CHECK: %[[result:.*]] = memref.cast %[[subview]] : memref<4x3x2xi64, strided<[6, 2, 1], offset: ?>> to memref<4x3x2xi64>
// CHECK: %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<4x3x2xi64> to !modelica.array<4x3x2x!modelica.int>
// CHECK: return %[[result_cast]] : !modelica.array<4x3x2x!modelica.int>

func.func @foo(%arg0: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<4x3x2x!modelica.int> {
    %0 = arith.constant 3 : index
    %1 = arith.constant 2 : index
    %2 = modelica.subscription %arg0[%0, %1] : !modelica.array<6x5x4x3x2x!modelica.int>, index, index -> !modelica.array<4x3x2x!modelica.int>
    func.return %2 : !modelica.array<4x3x2x!modelica.int>
}

// -----

// Slicing.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<?x?x4x3x2x!modelica.int>
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<6x5x4x3x2x!modelica.int> to memref<6x5x4x3x2xi64>
// CHECK-DAG: %[[int_range:.*]] = modelica.constant #modelica.int_range<3, 10, 2>
// CHECK-DAG: %[[int_range_begin:.*]] = modelica.iterable_begin %[[int_range]]
// CHECK-DAG: %[[int_range_begin_casted:.*]] = modelica.cast %[[int_range_begin]] : !modelica.int -> index
// CHECK-DAG: %[[int_range_step:.*]] = modelica.iterable_step %[[int_range]]
// CHECK-DAG: %[[int_range_step_casted:.*]] = modelica.cast %[[int_range_step]] : !modelica.int -> index
// CHECK-DAG: %[[int_range_size:.*]] = modelica.iterable_size %[[int_range]]
// CHECK-DAG: %[[real_range:.*]] = modelica.constant #modelica.real_range<3.000000e+00, 1.000000e+01, 2.000000e+00>
// CHECK-DAG: %[[real_range_begin:.*]] = modelica.iterable_begin %[[real_range]]
// CHECK-DAG: %[[real_range_begin_casted:.*]] = modelica.cast %[[real_range_begin]] : !modelica.real -> index
// CHECK-DAG: %[[real_range_step:.*]] = modelica.iterable_step %[[real_range]]
// CHECK-DAG: %[[real_range_step_casted:.*]] = modelica.cast %[[real_range_step]] : !modelica.real -> index
// CHECK-DAG: %[[real_range_size:.*]] = modelica.iterable_size %[[real_range]]
// CHECK: %[[subview:.*]] = memref.subview %[[memref]][%[[int_range_begin_casted]], %[[real_range_begin_casted]], 0, 0, 0] [%[[int_range_size]], %[[real_range_size]], 4, 3, 2] [%[[int_range_step_casted]], %[[real_range_step_casted]], 1, 1, 1] : memref<6x5x4x3x2xi64> to memref<?x?x4x3x2xi64, strided<[?, ?, 6, 2, 1], offset: ?>>
// CHECK: %[[result:.*]] = memref.cast %[[subview]] : memref<?x?x4x3x2xi64, strided<[?, ?, 6, 2, 1], offset: ?>> to memref<?x?x4x3x2xi64>
// CHECK: %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<?x?x4x3x2xi64> to !modelica.array<?x?x4x3x2x!modelica.int>
// CHECK: return %[[result_cast]] : !modelica.array<?x?x4x3x2x!modelica.int>

func.func @foo(%arg0: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<?x?x4x3x2x!modelica.int> {
    %0 = modelica.constant #modelica.int_range<3, 10, 2> : !modelica<iterable !modelica.int>
    %1 = modelica.constant #modelica.real_range<3.0, 10.0, 2.0> : !modelica<iterable !modelica.real>
    %2 = modelica.subscription %arg0[%0, %1] : !modelica.array<6x5x4x3x2x!modelica.int>, !modelica<iterable !modelica.int>, !modelica<iterable !modelica.real> -> !modelica.array<?x?x4x3x2x!modelica.int>
    func.return %2 : !modelica.array<?x?x4x3x2x!modelica.int>
}
