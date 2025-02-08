// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @TanTest0

func.func @TanTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.tan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @TanTest1

func.func @TanTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.523598775>
    %result = bmodelica.tan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.577350
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @TanTest2

func.func @TanTest2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.785398163>
    %result = bmodelica.tan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.999999
    // CHECK: return %[[cst]]
}
