// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer

func.func @Integer() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.sub %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real

func.func @Real() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.sub %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerReal

func.func @IntegerReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.sub %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealInteger

func.func @RealInteger() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.sub %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirIndex

func.func @mlirIndex() -> (index) {
    %x = bmodelica.constant 3 : index
    %y = bmodelica.constant 2 : index
    %result = bmodelica.sub %x, %y : (index, index) -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirInteger

func.func @mlirInteger() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %y = bmodelica.constant 2 : i64
    %result = bmodelica.sub %x, %y : (i64, i64) -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @mlirFloat

func.func @mlirFloat() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %y = bmodelica.constant 2.0 : f64
    %result = bmodelica.sub %x, %y : (f64, f64) -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1.000000e+00 : f64
    // CHECK: return %[[cst]]
}
