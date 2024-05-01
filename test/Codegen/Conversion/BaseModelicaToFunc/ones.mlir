// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK: func.func private @_Mones_void_ai64(memref<*xi64>)

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> !bmodelica.array<?x?x!bmodelica.int>
// CHECK-DAG: %[[result:.*]] = bmodelica.alloc %[[arg0]], %[[arg1]] : <?x?x!bmodelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.array<?x?x!bmodelica.int> to memref<?x?xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<?x?xi64> to memref<*xi64>
// CHECK: call @_Mones_void_ai64(%[[result_unranked]]) : (memref<*xi64>) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: index, %arg1: index) -> !bmodelica.array<?x?x!bmodelica.int> {
    %0 = bmodelica.ones %arg0, %arg1 : (index, index) -> !bmodelica.array<?x?x!bmodelica.int>
    func.return %0 : !bmodelica.array<?x?x!bmodelica.int>
}
