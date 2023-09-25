// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[result:.*]] = modelica.constant 4 : index
// CHECK: return %[[result]]

func.func @test() -> index {
    %0 = modelica.constant #modelica.int_range<3, 10, 2>
    %1 = modelica.range_size %0 : !modelica<range !modelica.int>
    return %1 : index
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[result:.*]] = modelica.constant 3 : index
// CHECK: return %[[result]]

func.func @test() -> index {
    %0 = modelica.constant #modelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
    %1 = modelica.range_size %0 : !modelica<range !modelica.real>
    return %1 : index
}
