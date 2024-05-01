// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK: func.func private @_Minteger_f64_f64(f64) -> f64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Minteger_f64_f64(%[[arg0_casted]]) : (f64) -> f64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.integer %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}
