// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Boolean

func.func @Boolean() -> !bmodelica.bool {
    %0 = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
    return %0 : !bmodelica.bool

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
    // CHECK-NEXT: return %[[cst]]
}

// -----

// CHECK-LABEL: @Integer

func.func @Integer() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica<int 42> : !bmodelica.int
    return %0 : !bmodelica.int

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 42> : !bmodelica.int
    // CHECK-NEXT: return %[[cst]]
}

// -----

// CHECK-LABEL: @Real

func.func @Real() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica<real 3.14> : !bmodelica.real
    return %0 : !bmodelica.real

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.140000e+00> : !bmodelica.real
    // CHECK-NEXT: return %[[cst]]
}

// -----

// CHECK-LABEL: @i64

func.func @i64() -> i64 {
    %0 = bmodelica.constant 42 : i64
    return %0 : i64

    // CHECK: %[[cst:.*]] = bmodelica.constant 42 : i64
    // CHECK-NEXT: return %[[cst]] : i64
}

// -----

// CHECK-LABEL: @f64

func.func @f64() -> f64 {
    %0 = bmodelica.constant 3.14 : f64
    return %0 : f64

    // CHECK: %[[cst:.*]] = bmodelica.constant 3.140000e+00 : f64
    // CHECK-NEXT: return %[[cst]] : f64
}
