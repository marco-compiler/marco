// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = bmodelica.constant #bmodelica.int_range<3, 10, 2>
// CHECK: return %[[range]] : !bmodelica<range !bmodelica.int>

func.func @test() -> !bmodelica<range !bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.int<3>
    %1 = bmodelica.constant #bmodelica.int<10>
    %2 = bmodelica.constant #bmodelica.int<2>
    %3 = bmodelica.range %0, %1, %2 : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> !bmodelica<range !bmodelica.int>
    return  %3 : !bmodelica<range !bmodelica.int>
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = bmodelica.constant #bmodelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
// CHECK: return %[[range]] : !bmodelica<range !bmodelica.real>

func.func @test() -> !bmodelica<range !bmodelica.real> {
    %0 = bmodelica.constant #bmodelica.real<3.000000e+00>
    %1 = bmodelica.constant #bmodelica.real<1.000000e+01>
    %2 = bmodelica.constant #bmodelica.real<2.500000e+00>
    %3 = bmodelica.range %0, %1, %2 : (!bmodelica.real, !bmodelica.real, !bmodelica.real) -> !bmodelica<range !bmodelica.real>
    return  %3 : !bmodelica<range !bmodelica.real>
}

// -----

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = bmodelica.constant #bmodelica.int_range<3, 10, 2>
// CHECK: return %[[range]] : !bmodelica<range index>

func.func @test() -> !bmodelica<range index> {
    %0 = bmodelica.constant 3 : index
    %1 = bmodelica.constant 10 : index
    %2 = bmodelica.constant 2 : index
    %3 = bmodelica.range %0, %1, %2 : (index, index, index) -> !bmodelica<range index>
    return  %3 : !bmodelica<range index>
}
