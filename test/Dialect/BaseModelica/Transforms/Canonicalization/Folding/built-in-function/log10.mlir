// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerTest0

func.func @IntegerTest0() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 1>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerTest1

func.func @IntegerTest1() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerTest2

func.func @IntegerTest2() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 100>
    %result = bmodelica.log10 %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealTest0

func.func @RealTest0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealTest1

func.func @RealTest1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealTest2

func.func @RealTest2() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 100.0>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealTest3

func.func @RealTest3() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.1>
    %result = bmodelica.log10 %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -1.000000e+00>
    // CHECK: return %[[cst]]
}
