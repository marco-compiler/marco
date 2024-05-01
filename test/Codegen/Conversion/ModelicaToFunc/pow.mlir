// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: func.func private @_Mpow_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Mpow_f64_f64_f64(%[[arg0_casted]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.pow %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}
