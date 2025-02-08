// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerPositiveStep

func.func @IntegerPositiveStep() -> index {
    %0 = bmodelica.constant #bmodelica.int_range<3, 10, 2>
    %1 = bmodelica.range_size %0 : !bmodelica<range !bmodelica.int>
    return %1 : index

    // CHECK: %[[cst:.*]] = bmodelica.constant 4 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealPositiveStep

func.func @RealPositiveStep() -> index {
    %0 = bmodelica.constant #bmodelica.real_range<3.000000e+00, 1.000000e+01, 2.500000e+00>
    %1 = bmodelica.range_size %0 : !bmodelica<range !bmodelica.real>
    return %1 : index

    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerNegativeStep

func.func @IntegerNegativeStep() -> index {
    %0 = bmodelica.constant #bmodelica.int_range<10, 3, -2>
    %1 = bmodelica.range_size %0 : !bmodelica<range !bmodelica.int>
    return %1 : index

    // CHECK: %[[cst:.*]] = bmodelica.constant 4 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealNegativeStep

func.func @RealNegativeStep() -> index {
    %0 = bmodelica.constant #bmodelica.real_range<1.000000e+01, 3.000000e+00, -2.500000e+00>
    %1 = bmodelica.range_size %0 : !bmodelica<range !bmodelica.real>
    return %1 : index

    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}
