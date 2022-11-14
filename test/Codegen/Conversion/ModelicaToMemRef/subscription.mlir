// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

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
    %2 = modelica.subscription %arg0[%0, %1] : !modelica.array<6x5x4x3x2x!modelica.int>
    func.return %2 : !modelica.array<4x3x2x!modelica.int>
}
