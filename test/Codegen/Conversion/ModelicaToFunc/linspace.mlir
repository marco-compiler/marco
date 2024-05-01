// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: func.func private @_Mlinspace_void_ai64_f64_f64(memref<*xi64>, f64, f64)

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real, %[[arg2:.*]]: index) -> !bmodelica.array<?x!bmodelica.int>
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK-DAG: %[[result:.*]] = bmodelica.alloc %[[arg2]] : <?x!bmodelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.array<?x!bmodelica.int> to memref<?xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<?xi64> to memref<*xi64>
// CHECK: call @_Mlinspace_void_ai64_f64_f64(%[[result_unranked]], %[[arg0_casted]], %[[arg1_casted]]) : (memref<*xi64>, f64, f64) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: !bmodelica.real, %arg1: !bmodelica.real, %arg2: index) -> !bmodelica.array<?x!bmodelica.int> {
    %0 = bmodelica.linspace %arg0, %arg1, %arg2 : (!bmodelica.real, !bmodelica.real, index) -> !bmodelica.array<?x!bmodelica.int>
    func.return %0 : !bmodelica.array<?x!bmodelica.int>
}
