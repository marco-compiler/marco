// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>)
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK: memref.dealloc %[[memref]] : memref<5x3xi64>
// CHECK: return

func.func @foo(%arg0: !modelica.array<5x3x!modelica.int>) {
    modelica.free %arg0 : !modelica.array<5x3x!modelica.int>
    func.return
}
