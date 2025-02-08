// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Test0

func.func @Test0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.cos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test1

func.func @Test1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.523598775>
    %result = bmodelica.cos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.866025
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test2

func.func @Test2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.785398163>
    %result = bmodelica.cos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.707106
    // CHECK: return %[[cst]]
}
