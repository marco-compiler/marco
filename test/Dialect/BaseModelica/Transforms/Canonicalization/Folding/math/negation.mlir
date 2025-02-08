// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer

func.func @Integer() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.neg %x: !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int -3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real

func.func @Real() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.neg %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirIndex

func.func @mlirIndex() -> (index) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.neg %x : index -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant -3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirInteger

func.func @mlirInteger() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.neg %x : i64 -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant -3 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirFloat

func.func @mlirFloat() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.neg %x : f64 -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant -3.000000e+00 : f64
    // CHECK: return %[[cst]]
}
