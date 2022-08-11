// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @sum : (memref<*xi64>) -> i64
// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK: %[[arg0_unranked:.*]] = memref.cast %[[arg0_casted]] : memref<5x3xi64> to memref<*xi64>
// CHECK: %[[result:.*]] = modelica.call @sum(%[[arg0_unranked]]) : (memref<*xi64>) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.int {
    %0 = modelica.sum %arg0 : !modelica.array<5x3x!modelica.int> -> !modelica.int
    func.return %0 : !modelica.int
}
