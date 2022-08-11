// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK: modelica.runtime_function @sign : (f64) -> i64

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real) -> !modelica.int
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK: %[[result:.*]] = modelica.call @sign(%[[arg0_casted]]) : (f64) -> i64
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @test(%arg0: !modelica.real) -> !modelica.int {
    %0 = modelica.sign %arg0 : !modelica.real -> !modelica.int
    func.return %0 : !modelica.int
}
