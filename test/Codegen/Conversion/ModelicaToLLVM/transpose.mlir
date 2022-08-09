// RUN: modelica-opt %s --split-input-file --convert-modelica-to-llvm | FileCheck %s

// CHECK-LABEL: @staticArray()
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.array<3x5x!modelica.int>
// CHECK-DAG: %[[operand_cast:.*]] = builtin.unrealized_conversion_cast %[[operand]] : !modelica.array<5x3x!modelica.int> to !modelica.array<5x3x!modelica.int>
// CHECK: %[[operand:.*]] = modelica.alloc : !modelica.array<5x3x!modelica.int>

// CHECK-DAG: %[[result:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK: %[[result_cast1:.*]] = modelica.array_cast %[[result]] : !modelica.array<3x5x!modelica.int> -> !modelica.array<*x!modelica.int>
// CHECK: %[[result_cast2:.*]] = builtin.unrealized_conversion_cast %[[result_cast1]] : !modelica.array<*x!modelica.int> to !llvm.struct<(i64, ptr<i8>)>

// CHECK: %[[memref:.*]] = memref.alloc() : memref<5x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[memref]] : memref<5x3xi64> to !modelica.array<5x3x!modelica.int>
// CHECK: return %[[result]]

func.func @staticArray(%arg0 : !modelica.array<5x3x!modelica.int>) -> !modelica.array<3x5x!modelica.int> {
    %0 = modelica.transpose %arg0 : !modelica.array<5x3x!modelica.int> -> !modelica.array<3x5x!modelica.int>
    func.return %0 : !modelica.array<3x5x!modelica.int>
}
