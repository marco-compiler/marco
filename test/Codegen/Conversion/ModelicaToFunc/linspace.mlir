// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: func.func private @_Mlinspace_void_ai64_f64_f64(memref<*xi64>, f64, f64)

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real, %[[arg2:.*]]: index) -> !modelica.array<?x!modelica.int>
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK-DAG: %[[result:.*]] = modelica.alloc %[[arg2]] : <?x!modelica.int>
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.array<?x!modelica.int> to memref<?xi64>
// CHECK-DAG: %[[result_unranked:.*]] = memref.cast %[[result_casted]] : memref<?xi64> to memref<*xi64>
// CHECK: call @_Mlinspace_void_ai64_f64_f64(%[[result_unranked]], %[[arg0_casted]], %[[arg1_casted]]) : (memref<*xi64>, f64, f64) -> ()
// CHECK: return %[[result]]

func.func @test(%arg0: !modelica.real, %arg1: !modelica.real, %arg2: index) -> !modelica.array<?x!modelica.int> {
    %0 = modelica.linspace %arg0, %arg1, %arg2 : (!modelica.real, !modelica.real, index) -> !modelica.array<?x!modelica.int>
    func.return %0 : !modelica.array<?x!modelica.int>
}
