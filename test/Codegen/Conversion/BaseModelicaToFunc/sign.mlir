// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK: func.func private @_Msign_i64_f64(f64) -> i64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.int
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = call @_Msign_i64_f64(%[[arg0_casted]]) : (f64) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !bmodelica.real) -> !bmodelica.int {
    %0 = bmodelica.sign %arg0 : !bmodelica.real -> !bmodelica.int
    func.return %0 : !bmodelica.int
}
