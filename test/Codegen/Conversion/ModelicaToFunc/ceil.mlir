// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @_Mceil_f64_f64 : (f64) -> f64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real) -> !modelica.real
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @_Mceil_f64_f64(%[[arg0_casted]]) : (f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !modelica.real) -> !modelica.real {
    %0 = modelica.ceil %arg0 : !modelica.real -> !modelica.real
    func.return %0 : !modelica.real
}
