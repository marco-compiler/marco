// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @SinhTest0

func.func @SinhTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -1.0>
    %result = bmodelica.sinh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -1.175201
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @SinhTest1

func.func @SinhTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.sinh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @SinhTest2

func.func @SinhTest2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.sinh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.175201
    // CHECK: return %[[cst]]
}
