// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @emptyShape
// CHECK: %[[memref:.*]] = memref.alloc() : memref<i64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<i64> to !bmodelica.array<!bmodelica.int>
// CHECK: return %[[result]]

func.func @emptyShape() -> !bmodelica.array<!bmodelica.int> {
    %0 = bmodelica.alloc : <!bmodelica.int>
    func.return %0 : !bmodelica.array<!bmodelica.int>
}

// -----

// CHECK-LABEL: @fixedSize
// CHECK: %[[memref:.*]] = memref.alloc() : memref<5x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<5x3xi64> to !bmodelica.array<5x3x!bmodelica.int>
// CHECK: return %[[result]]

func.func @fixedSize() -> !bmodelica.array<5x3x!bmodelica.int> {
    %0 = bmodelica.alloc : <5x3x!bmodelica.int>
    func.return %0 : !bmodelica.array<5x3x!bmodelica.int>
}

// -----

// CHECK-LABEL: @dynamicSize
// CHECK: %[[size:.*]] = arith.constant 3 : index
// CHECK: %[[memref:.*]] = memref.alloc(%[[size]]) : memref<5x?xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<5x?xi64> to !bmodelica.array<5x?x!bmodelica.int>
// CHECK: return %[[result]]

func.func @dynamicSize() -> !bmodelica.array<5x?x!bmodelica.int> {
    %0 = arith.constant 3 : index
    %1 = bmodelica.alloc %0 : <5x?x!bmodelica.int>
    func.return %1 : !bmodelica.array<5x?x!bmodelica.int>
}
