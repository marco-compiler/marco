// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer

func.func @Integer() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica.int_range<0, 5, 1>
    %1 = bmodelica.range_begin %0 : !bmodelica<range !bmodelica.int> -> !bmodelica.int
    return %1 : !bmodelica.int

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real

func.func @Real() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
    %1 = bmodelica.range_begin %0 : !bmodelica<range !bmodelica.real> -> !bmodelica.real
    return %1 : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}
