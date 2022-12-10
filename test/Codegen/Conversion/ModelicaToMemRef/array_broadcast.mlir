// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @fixedSize
// CHECK:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK:       %[[value_casted:.*]] = builtin.unrealized_conversion_cast %[[value]] : !modelica.int to i64
// CHECK:       %[[memref:.*]] = memref.alloc() : memref<3xi64>

// CHECK:       %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[one:.*]] = arith.constant 1 : index

// CHECK:       %[[dim0_index:.*]] = arith.constant 0 : index
// CHECK:       %[[dim0:.*]] = memref.dim %[[memref]], %[[dim0_index]]
// CHECK:       scf.for %[[i0:.*]] = %[[zero]] to %[[dim0]] step %[[one]] {
// CHECK-NEXT:      memref.store %[[value_casted]], %[[memref]][%[[i0]]]
// CHECK-NEXT:  }

func.func @fixedSize() -> !modelica.array<3x!modelica.int> {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x!modelica.int>
    func.return %1 : !modelica.array<3x!modelica.int>
}

// -----

// CHECK-LABEL: @dynamicSize
// CHECK:       %[[dynamicDim0:.*]] = arith.constant 2 : index
// CHECK:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK:       %[[value_casted:.*]] = builtin.unrealized_conversion_cast %[[value]] : !modelica.int to i64
// CHECK:       %[[memref:.*]] = memref.alloc(%[[dynamicDim0]]) : memref<?xi64>

// CHECK:       %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[one:.*]] = arith.constant 1 : index

// CHECK:       %[[dim0_index:.*]] = arith.constant 0 : index
// CHECK:       %[[dim0:.*]] = memref.dim %[[memref]], %[[dim0_index]]
// CHECK:       scf.for %[[i0:.*]] = %[[zero]] to %[[dim0]] step %[[one]] {
// CHECK-NEXT:      memref.store %[[value_casted]], %[[memref]][%[[i0]]]
// CHECK-NEXT:  }

func.func @dynamicSize() -> !modelica.array<?x!modelica.int> {
    %0 = arith.constant 2 : index
    %1 = modelica.constant #modelica.int<0>
    %2 = modelica.array_broadcast %1, %0 : !modelica.int -> !modelica.array<?x!modelica.int>
    func.return %2 : !modelica.array<?x!modelica.int>
}

// -----

// CHECK-LABEL: @multidimensionalArray
// CHECK: %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK: %[[value_casted:.*]] = builtin.unrealized_conversion_cast %[[value]] : !modelica.int to i64
// CHECK: %[[memref:.*]] = memref.alloc() : memref<2x3x4xi64>

// CHECK: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[one:.*]] = arith.constant 1 : index

// CHECK:       %[[dim0_index:.*]] = arith.constant 0 : index
// CHECK:       %[[dim0:.*]] = memref.dim %[[memref]], %[[dim0_index]]
// CHECK:       %[[dim1_index:.*]] = arith.constant 1 : index
// CHECK:       %[[dim1:.*]] = memref.dim %[[memref]], %[[dim1_index]]
// CHECK:       %[[dim2_index:.*]] = arith.constant 2 : index
// CHECK:       %[[dim2:.*]] = memref.dim %[[memref]], %[[dim2_index]]
// CHECK:       scf.for %[[i0:.*]] = %[[zero]] to %[[dim0]] step %[[one]] {
// CHECK:           scf.for %[[i1:.*]] = %[[zero]] to %[[dim1]] step %[[one]] {
// CHECK:               scf.for %[[i2:.*]] = %[[zero]] to %[[dim2]] step %[[one]] {
// CHECK-NEXT:              memref.store %[[value_casted]], %[[memref]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

func.func @multidimensionalArray() -> !modelica.array<2x3x4x!modelica.int> {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<2x3x4x!modelica.int>
    func.return %1 : !modelica.array<2x3x4x!modelica.int>
}

// -----

// CHECK-LABEL: @implicitCast
// CHECK:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK:       %[[memref:.*]] = memref.alloc() : memref<3xf64>

// CHECK:       %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[one:.*]] = arith.constant 1 : index

// CHECK:       %[[dim0_index:.*]] = arith.constant 0 : index
// CHECK:       %[[dim0:.*]] = memref.dim %[[memref]], %[[dim0_index]]

// CHECK:       %[[value_real:.*]] = modelica.cast %[[value]] : !modelica.int -> !modelica.real
// CHECK:       %[[value_casted:.*]] = builtin.unrealized_conversion_cast %[[value_real]] : !modelica.real to f64

// CHECK:       scf.for %[[i0:.*]] = %[[zero]] to %[[dim0]] step %[[one]] {
// CHECK-NEXT:      memref.store %[[value_casted]], %[[memref]][%[[i0]]]
// CHECK-NEXT:  }

func.func @implicitCast() -> !modelica.array<3x!modelica.real> {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x!modelica.real>
    func.return %1 : !modelica.array<3x!modelica.real>
}
