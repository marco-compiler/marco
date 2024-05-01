// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK: func.func private @_Mdiagonal_void_ai64_ai64(memref<*xi64>, memref<*xi64>)

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x!bmodelica.int>) -> !bmodelica.array<3x3x!bmodelica.int>
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<3x!bmodelica.int> to memref<3xi64>
// CHECK-DAG: %[[result:.*]] = bmodelica.alloc : <3x3x!bmodelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.array<3x3x!bmodelica.int> to memref<3x3xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<3x3xi64> to memref<*xi64>
// CHECK-DAG: %[[arg0_unranked:.*]] = memref.cast %[[arg0_casted]] : memref<3xi64> to memref<*xi64>
// CHECK: call @_Mdiagonal_void_ai64_ai64(%[[result_unranked]], %[[arg0_unranked]]) : (memref<*xi64>, memref<*xi64>) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: !bmodelica.array<3x!bmodelica.int>) -> !bmodelica.array<3x3x!bmodelica.int> {
    %0 = bmodelica.diagonal %arg0 : !bmodelica.array<3x!bmodelica.int> -> !bmodelica.array<3x3x!bmodelica.int>
    func.return %0 : !bmodelica.array<3x3x!bmodelica.int>
}
