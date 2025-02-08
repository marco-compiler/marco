// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @ZeroExponent

func.func @ZeroExponent() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @OneExponent

func.func @OneExponent() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.718281
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @PositiveExponent

func.func @PositiveExponent() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 7.389056
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @NegativeExponent

func.func @NegativeExponent() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -2.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.135335
    // CHECK: return %[[cst]]
}
