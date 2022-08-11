// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @transpose : (memref<*xi64>, memref<*xi64>) -> ()

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.array<3x5x!modelica.int>
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK-DAG: %[[result:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<3x5xi64> to memref<*xi64>
// CHECK-DAG: %[[arg0_unranked:.*]] = memref.cast %[[arg0_casted]] : memref<5x3xi64> to memref<*xi64>
// CHECK: modelica.call @transpose(%[[result_unranked]], %[[arg0_unranked]]) : (memref<*xi64>, memref<*xi64>) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.array<3x5x!modelica.int> {
    %0 = modelica.transpose %arg0 : !modelica.array<5x3x!modelica.int> -> !modelica.array<3x5x!modelica.int>
    func.return %0 : !modelica.array<3x5x!modelica.int>
}
