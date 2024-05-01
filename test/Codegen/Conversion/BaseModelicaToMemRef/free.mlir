// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-memref | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>)
// CHECK: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK: memref.dealloc %[[memref]] : memref<5x3xi64>
// CHECK: return

func.func @foo(%arg0: !bmodelica.array<5x3x!bmodelica.int>) {
    bmodelica.free %arg0 : !bmodelica.array<5x3x!bmodelica.int>
    func.return
}
