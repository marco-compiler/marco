// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x?x!modelica.int>) -> index
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x?x!modelica.int> to memref<5x?xi64>
// CHECK: %[[dimension:.*]] = arith.constant 1 : index
// CHECK: %[[result:.*]] = memref.dim %[[memref]], %[[dimension]] : memref<5x?xi64>
// CHECK: return %[[result]] : index

func.func @foo(%arg0: !modelica.array<5x?x!modelica.int>) -> index {
    %0 = arith.constant 1 : index
    %1 = modelica.dim %arg0, %0 : !modelica.array<5x?x!modelica.int>
    func.return %1 : index
}
