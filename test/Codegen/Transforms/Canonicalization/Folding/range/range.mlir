// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = modelica.constant #modelica.int_range<3, 10, 2>
// CHECK: return %[[range]] : !modelica<range !modelica.int>

func.func @test() -> !modelica<range !modelica.int> {
    %0 = modelica.constant #modelica.int<3>
    %1 = modelica.constant #modelica.int<10>
    %2 = modelica.constant #modelica.int<2>
    %3 = modelica.range %0, %1, %2 : (!modelica.int, !modelica.int, !modelica.int) -> !modelica<range !modelica.int>
    return  %3 : !modelica<range !modelica.int>
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = modelica.constant #modelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
// CHECK: return %[[range]] : !modelica<range !modelica.real>

func.func @test() -> !modelica<range !modelica.real> {
    %0 = modelica.constant #modelica.real<3.000000e+00>
    %1 = modelica.constant #modelica.real<1.000000e+01>
    %2 = modelica.constant #modelica.real<2.500000e+00>
    %3 = modelica.range %0, %1, %2 : (!modelica.real, !modelica.real, !modelica.real) -> !modelica<range !modelica.real>
    return  %3 : !modelica<range !modelica.real>
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = modelica.constant #modelica.int_range<3, 10, 2>
// CHECK: return %[[range]] : !modelica<range index>

func.func @test() -> !modelica<range index> {
    %0 = modelica.constant 3 : index
    %1 = modelica.constant 10 : index
    %2 = modelica.constant 2 : index
    %3 = modelica.range %0, %1, %2 : (index, index, index) -> !modelica<range index>
    return  %3 : !modelica<range index>
}
