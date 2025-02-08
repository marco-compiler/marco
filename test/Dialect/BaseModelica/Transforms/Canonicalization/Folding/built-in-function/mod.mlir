// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerZeroRemainder

func.func @IntegerZeroRemainder() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 6>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.mod %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerNonZeroRemainder

func.func @IntegerNonZeroRemainder() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 8>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.mod %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerNegativeDivisor

func.func @IntegerNegativeDivisor() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int -3>
    %result = bmodelica.mod %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int -2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerNegativeDividend

func.func @IntegerNegativeDividend() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int -10>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.mod %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealZeroRemainder

func.func @RealZeroRemainder() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 6.0>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealNonZeroRemainder0

func.func @RealNonZeroRemainder0() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 8.5>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealNonZeroRemainder1

func.func @RealNonZeroRemainder1() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 10.5>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealNegativeDividend

func.func @RealNegativeDividend() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -10.5>
    %y = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealNegativeDivisor

func.func @RealNegativeDivisor() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 10.5>
    %y = bmodelica.constant #bmodelica<real -3.0>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -1.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealInteger

func.func @RealInteger() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 8.5>
    %y = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.mod %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerReal

func.func @IntegerReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 3.75>
    %result = bmodelica.mod %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 2.500000e+00>
    // CHECK: return %[[cst]]
}
