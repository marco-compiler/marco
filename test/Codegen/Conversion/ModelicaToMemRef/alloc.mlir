// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @fixedSize
// CHECK: %[[memref:.*]] = memref.alloc() : memref<5x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<5x3xi64> to !modelica.array<5x3x!modelica.int>
// CHECK: return %[[result]]

func.func @fixedSize() -> !modelica.array<5x3x!modelica.int> {
    %0 = modelica.alloc : !modelica.array<5x3x!modelica.int>
    func.return %0 : !modelica.array<5x3x!modelica.int>
}

// -----

// CHECK-LABEL: @dynamicSize
// CHECK: %[[size:.*]] = arith.constant 3 : index
// CHECK: %[[memref:.*]] = memref.alloc(%[[size]]) : memref<5x?xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<5x?xi64> to !modelica.array<5x?x!modelica.int>
// CHECK: return %[[result]]

func.func @dynamicSize() -> !modelica.array<5x?x!modelica.int> {
    %0 = arith.constant 3 : index
    %1 = modelica.alloc %0 : !modelica.array<5x?x!modelica.int>
    func.return %1 : !modelica.array<5x?x!modelica.int>
}
