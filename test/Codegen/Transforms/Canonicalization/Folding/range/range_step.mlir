// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[result:.*]] = modelica.constant #modelica.int<1>
// CHECK: return %[[result]]

func.func @test() -> !modelica.int {
    %0 = modelica.constant #modelica.int_range<0, 5, 1>
    %1 = modelica.range_step %0 : !modelica<range !modelica.int> -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[result:.*]] = modelica.constant #modelica.real<2.500000e+00>
// CHECK: return %[[result]]

func.func @test() -> !modelica.real {
    %0 = modelica.constant #modelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
    %1 = modelica.range_step %0 : !modelica<range !modelica.real> -> !modelica.real
    return %1 : !modelica.real
}
