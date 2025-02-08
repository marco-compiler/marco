// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @SinTest0

func.func @SinTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.sin %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @SinTest1

func.func @SinTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.523598775>
    %result = bmodelica.sin %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.499999
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @SinTest2

func.func @SinTest2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.785398163>
    %result = bmodelica.sin %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.707106
    // CHECK: return %[[cst]]
}
