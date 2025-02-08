// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @LogTest0

func.func @LogTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @LogTest1

func.func @LogTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.718281828>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.999999
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @LogTest2

func.func @LogTest2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 7.389056099>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @LogTest3

func.func @LogTest3() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.36787944>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -1.000000
    // CHECK: return %[[cst]]
}
