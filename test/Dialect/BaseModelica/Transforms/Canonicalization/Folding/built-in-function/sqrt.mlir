// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer0

func.func @Integer0() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Integer1

func.func @Integer1() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 1>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Integer4

func.func @Integer4() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 4>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Integer9

func.func @Integer9() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 9>
    %result = bmodelica.sqrt %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real0

func.func @Real0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real1

func.func @Real1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real4

func.func @Real4() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 4.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real9

func.func @Real9() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 9.0>
    %result = bmodelica.sqrt %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}
