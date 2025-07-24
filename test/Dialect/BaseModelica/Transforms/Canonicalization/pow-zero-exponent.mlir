// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer

func.func @Integer(%0: !bmodelica.int) -> (!bmodelica.int) {
    %cst = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.pow %0, %cst : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK: %[[result:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK: return %[[result]]
}

// -----

// CHECK-LABEL: @Real

func.func @Real(%0: !bmodelica.real) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.pow %0, %cst : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[result]]
}

// -----

// CHECK-LABEL: @IntegerReal

func.func @IntegerReal(%0: !bmodelica.int) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.pow %0, %cst : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[result]]
}

// -----

// CHECK-LABEL: @RealInteger

func.func @RealInteger(%0: !bmodelica.real) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.pow %0, %cst : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[result]]
}
