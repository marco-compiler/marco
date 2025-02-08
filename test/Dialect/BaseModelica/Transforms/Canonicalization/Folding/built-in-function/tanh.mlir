// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @TanhTest0

func.func @TanhTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.tanh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @TanhTest1

func.func @TanhTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.tanh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.761594
    // CHECK: return %[[cst]]
}
