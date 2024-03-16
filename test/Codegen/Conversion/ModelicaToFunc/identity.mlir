// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: func.func private @_Midentity_void_ai64(memref<*xi64>)

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: index) -> !modelica.array<?x?x!modelica.int>
// CHECK-DAG: %[[result:.*]] = modelica.alloc %[[arg0]], %[[arg0]] : <?x?x!modelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.array<?x?x!modelica.int> to memref<?x?xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<?x?xi64> to memref<*xi64>
// CHECK: call @_Midentity_void_ai64(%[[result_unranked]]) : (memref<*xi64>) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: index) -> !modelica.array<?x?x!modelica.int> {
    %0 = modelica.identity %arg0 : index -> !modelica.array<?x?x!modelica.int>
    func.return %0 : !modelica.array<?x?x!modelica.int>
}
