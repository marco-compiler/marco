// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2)[s0] -> (d0 * 6 + s0 + d1 * 2 + d2)>
// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<4x3x2x!modelica.int>
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<6x5x4x3x2x!modelica.int> to memref<6x5x4x3x2xi64>
// CHECK: %[[c3:.*]] = arith.constant 3 : index
// CHECK: %[[c2:.*]] = arith.constant 2 : index
// CHECK: %[[subview:.*]] = memref.subview %[[memref]][%[[c3]], %[[c2]], 0, 0, 0] [1, 1, 4, 3, 2] [1, 1, 1, 1, 1] : memref<6x5x4x3x2xi64> to memref<4x3x2xi64, #map>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[subview]] : memref<4x3x2xi64, #map> to !modelica.array<4x3x2x!modelica.int>
// CHECK: return %[[result]] : !modelica.array<4x3x2x!modelica.int>

func.func @foo(%arg0: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<4x3x2x!modelica.int> {
    %0 = arith.constant 3 : index
    %1 = arith.constant 2 : index
    %2 = modelica.subscription %arg0[%0, %1] : !modelica.array<6x5x4x3x2x!modelica.int>
    func.return %2 : !modelica.array<4x3x2x!modelica.int>
}
