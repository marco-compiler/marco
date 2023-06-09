// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @oneElement
// CHECK-SAME: (%[[v0:.*]]: !modelica.int)
// CHECK: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !modelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<1xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<1xi64> to !modelica.array<1x!modelica.int>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: return %[[result]]

func.func @oneElement(%arg0: !modelica.int) -> !modelica.array<1x!modelica.int> {
    %0 = modelica.array_from_elements %arg0 : !modelica.int -> !modelica.array<1x!modelica.int>
    func.return %0 : !modelica.array<1x!modelica.int>
}

// -----

// CHECK-LABEL: @twoElements
// CHECK-SAME: (%[[v0:.*]]: !modelica.int, %[[v1:.*]]: !modelica.int)
// CHECK-DAG: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !modelica.int to i64
// CHECK-DAG: %[[v1_casted:.*]] = builtin.unrealized_conversion_cast %[[v1]] : !modelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<2xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<2xi64> to !modelica.array<2x!modelica.int>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: %[[i1:.*]] = arith.constant 1 : index
// CHECK: memref.store %[[v1_casted]], %[[memref]][%[[i1]]]
// CHECK: return %[[result]]

func.func @twoElements(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.array<2x!modelica.int> {
    %2 = modelica.array_from_elements %arg0, %arg1 : !modelica.int, !modelica.int -> !modelica.array<2x!modelica.int>
    func.return %2 : !modelica.array<2x!modelica.int>
}

// -----

// CHECK-LABEL: @multidimensionalArray
// CHECK-SAME: (%[[v0:.*]]: !modelica.int, %[[v1:.*]]: !modelica.int, %[[v2:.*]]: !modelica.int, %[[v3:.*]]: !modelica.int, %[[v4:.*]]: !modelica.int, %[[v5:.*]]: !modelica.int)
// CHECK-DAG: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0]] : !modelica.int to i64
// CHECK-DAG: %[[v1_casted:.*]] = builtin.unrealized_conversion_cast %[[v1]] : !modelica.int to i64
// CHECK-DAG: %[[v2_casted:.*]] = builtin.unrealized_conversion_cast %[[v2]] : !modelica.int to i64
// CHECK-DAG: %[[v3_casted:.*]] = builtin.unrealized_conversion_cast %[[v3]] : !modelica.int to i64
// CHECK-DAG: %[[v4_casted:.*]] = builtin.unrealized_conversion_cast %[[v4]] : !modelica.int to i64
// CHECK: %[[v5_casted:.*]] = builtin.unrealized_conversion_cast %[[v5]] : !modelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<2x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<2x3xi64> to !modelica.array<2x3x!modelica.int>
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

func.func @multidimensionalArray(%arg0: !modelica.int, %arg1: !modelica.int, %arg2: !modelica.int, %arg3: !modelica.int, %arg4: !modelica.int, %arg5: !modelica.int) -> !modelica.array<2x3x!modelica.int> {
    %0 = modelica.array_from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !modelica.int, !modelica.int, !modelica.int, !modelica.int, !modelica.int, !modelica.int -> !modelica.array<2x3x!modelica.int>
    func.return %0 : !modelica.array<2x3x!modelica.int>
}

// -----

// CHECK-LABEL: @implicitCast
// CHECK-SAME: (%[[v0:.*]]: !modelica.int)
// CHECK: %[[memref:.*]] = memref.alloc() : memref<1xf64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<1xf64> to !modelica.array<1x!modelica.real>
// CHECK: %[[i0:.*]] = arith.constant 0 : index
// CHECK: %[[v0_real:.*]] = modelica.cast %[[v0]] : !modelica.int -> !modelica.real
// CHECK: %[[v0_casted:.*]] = builtin.unrealized_conversion_cast %[[v0_real]] : !modelica.real to f64
// CHECK: memref.store %[[v0_casted]], %[[memref]][%[[i0]]]
// CHECK: return %[[result]]

func.func @implicitCast(%arg0: !modelica.int) -> !modelica.array<1x!modelica.real> {
    %0 = modelica.array_from_elements %arg0 : !modelica.int -> !modelica.array<1x!modelica.real>
    func.return %0 : !modelica.array<1x!modelica.real>
}
