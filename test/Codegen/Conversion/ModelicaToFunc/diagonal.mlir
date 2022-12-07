// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @_Mdiagonal_void_ai64_ai64 : (memref<*xi64>, memref<*xi64>) -> ()

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<3x3x!modelica.int>
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<3x!modelica.int> to memref<3xi64>
// CHECK-DAG: %[[result:.*]] = modelica.alloc : !modelica.array<3x3x!modelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.array<3x3x!modelica.int> to memref<3x3xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<3x3xi64> to memref<*xi64>
// CHECK-DAG: %[[arg0_unranked:.*]] = memref.cast %[[arg0_casted]] : memref<3xi64> to memref<*xi64>
// CHECK: modelica.call @_Mdiagonal_void_ai64_ai64(%[[result_unranked]], %[[arg0_unranked]]) : (memref<*xi64>, memref<*xi64>) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: !modelica.array<3x!modelica.int>) -> !modelica.array<3x3x!modelica.int> {
    %0 = modelica.diagonal %arg0 : !modelica.array<3x!modelica.int> -> !modelica.array<3x3x!modelica.int>
    func.return %0 : !modelica.array<3x3x!modelica.int>
}
