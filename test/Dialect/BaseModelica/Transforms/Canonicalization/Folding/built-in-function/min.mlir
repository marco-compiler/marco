// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerFirst

func.func @IntegerFirst() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerSecond

func.func @IntegerSecond() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerEqual

func.func @IntegerEqual() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealFirst

func.func @RealFirst() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealSecond

func.func @RealSecond() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealEqual

func.func @RealEqual() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealFirst

func.func @IntegerRealFirst() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealSecond

func.func @IntegerRealSecond() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 3>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealEqual

func.func @IntegerRealEqual() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 2>
    %y = bmodelica.constant #bmodelica<real 2.0>
    %result = bmodelica.min %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerFirst

func.func @RealIntegerFirst() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerSecond

func.func @RealIntegerSecond() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerEqual

func.func @RealIntegerEqual() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.0>
    %y = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.min %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}
