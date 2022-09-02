// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @pow : (f64, f64) -> f64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @pow(%[[arg0_casted]], %[[arg1_casted]]) : (f64, f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !modelica.real, %arg1: !modelica.real) -> !modelica.real {
    %0 = modelica.pow %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    func.return %0 : !modelica.real
}
