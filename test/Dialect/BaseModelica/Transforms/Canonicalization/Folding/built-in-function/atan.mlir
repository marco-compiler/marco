// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Test0

func.func @Test0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.atan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.785398
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test1

func.func @Test1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.577350269>
    %result = bmodelica.atan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.523598
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test2

func.func @Test2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.atan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test3

func.func @Test3() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -0.577350269>
    %result = bmodelica.atan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -0.523598
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Test4

func.func @Test4() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -1.0>
    %result = bmodelica.atan %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -0.785398
    // CHECK: return %[[cst]]
}
