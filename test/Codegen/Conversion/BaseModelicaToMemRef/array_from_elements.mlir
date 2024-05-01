// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-memref | FileCheck %s

// CHECK-LABEL: @oneElement
// CHECK-SAME: (%[[v0:.*]]: !bmodelica.int)
// CHECK: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !bmodelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<1xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<1xi64> to !bmodelica.array<1x!bmodelica.int>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: return %[[result]]

func.func @oneElement(%arg0: !bmodelica.int) -> !bmodelica.array<1x!bmodelica.int> {
    %0 = bmodelica.array_from_elements %arg0 : !bmodelica.int -> !bmodelica.array<1x!bmodelica.int>
    func.return %0 : !bmodelica.array<1x!bmodelica.int>
}

// -----

// CHECK-LABEL: @twoElements
// CHECK-SAME: (%[[v0:.*]]: !bmodelica.int, %[[v1:.*]]: !bmodelica.int)
// CHECK-DAG: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !bmodelica.int to i64
// CHECK-DAG: %[[v1_casted:.*]] = builtin.unrealized_conversion_cast %[[v1]] : !bmodelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<2xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: %[[i1:.*]] = arith.constant 1 : index
// CHECK: memref.store %[[v1_casted]], %[[memref]][%[[i1]]]
// CHECK: return %[[result]]

func.func @twoElements(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.array<2x!bmodelica.int> {
    %2 = bmodelica.array_from_elements %arg0, %arg1 : !bmodelica.int, !bmodelica.int -> !bmodelica.array<2x!bmodelica.int>
    func.return %2 : !bmodelica.array<2x!bmodelica.int>
}

// -----

// CHECK-LABEL: @multidimensionalArray
// CHECK-SAME: (%[[v0:.*]]: !bmodelica.int, %[[v1:.*]]: !bmodelica.int, %[[v2:.*]]: !bmodelica.int, %[[v3:.*]]: !bmodelica.int, %[[v4:.*]]: !bmodelica.int, %[[v5:.*]]: !bmodelica.int)
// CHECK-DAG: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !bmodelica.int to i64
// CHECK-DAG: %[[v1_casted:.*]] = builtin.unrealized_conversion_cast %[[v1]] : !bmodelica.int to i64
// CHECK-DAG: %[[v2_casted:.*]] = builtin.unrealized_conversion_cast %[[v2]] : !bmodelica.int to i64
// CHECK-DAG: %[[v3_casted:.*]] = builtin.unrealized_conversion_cast %[[v3]] : !bmodelica.int to i64
// CHECK-DAG: %[[v4_casted:.*]] = builtin.unrealized_conversion_cast %[[v4]] : !bmodelica.int to i64
// CHECK: %[[v5_casted:.*]] = builtin.unrealized_conversion_cast %[[v5]] : !bmodelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<2x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<2x3xi64> to !bmodelica.array<2x3x!bmodelica.int>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: %[[i1:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]], %[[i1]]]
// CHECK: %[[i2:.*]] = arith.constant 0 : index
// CHECK: %[[i3:.*]] = arith.constant 1 : index
// CHECK: memref.store %[[v1_casted]], %[[memref]][%[[i2]], %[[i3]]]
// CHECK: %[[i4:.*]] = arith.constant 0 : index
// CHECK: %[[i5:.*]] = arith.constant 2 : index
// CHECK: memref.store %[[v2_casted]], %[[memref]][%[[i4]], %[[i5]]]
// CHECK: %[[i6:.*]] = arith.constant 1 : index
// CHECK: %[[i7:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v3_casted]], %[[memref]][%[[i6]], %[[i7]]]
// CHECK: %[[i8:.*]] = arith.constant 1 : index
// CHECK: %[[i9:.*]] = arith.constant 1 : index
// CHECK: memref.store %[[v4_casted]], %[[memref]][%[[i8]], %[[i9]]]
// CHECK: %[[i10:.*]] = arith.constant 1 : index
// CHECK: %[[i11:.*]] = arith.constant 2 : index
// CHECK: memref.store %[[v5_casted]], %[[memref]][%[[i10]], %[[i11]]]
// CHECK: return %[[result]]

func.func @multidimensionalArray(%arg0: !bmodelica.int, %arg1: !bmodelica.int, %arg2: !bmodelica.int, %arg3: !bmodelica.int, %arg4: !bmodelica.int, %arg5: !bmodelica.int) -> !bmodelica.array<2x3x!bmodelica.int> {
    %0 = bmodelica.array_from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int -> !bmodelica.array<2x3x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x3x!bmodelica.int>
}

// -----

// CHECK-LABEL: @implicitCast
// CHECK-SAME: (%[[v0:.*]]: !bmodelica.int)
// CHECK: %[[memref:.*]] = memref.alloc() : memref<1xf64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<1xf64> to !bmodelica.array<1x!bmodelica.real>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: %[[v0_real:.*]] = bmodelica.cast %[[v0]] : !bmodelica.int -> !bmodelica.real
// CHECK: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0_real]] : !bmodelica.real to f64
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: return %[[result]]

func.func @implicitCast(%arg0: !bmodelica.int) -> !bmodelica.array<1x!bmodelica.real> {
    %0 = bmodelica.array_from_elements %arg0 : !bmodelica.int -> !bmodelica.array<1x!bmodelica.real>
    func.return %0 : !bmodelica.array<1x!bmodelica.real>
}
