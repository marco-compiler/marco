// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerFirst

func.func @IntegerFirst() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 9>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerEqual

func.func @IntegerEqual() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerSecond

func.func @IntegerSecond() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 9>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealFirst

func.func @RealFirst() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 9.0>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealEqual

func.func @RealEqual() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealSecond

func.func @RealSecond() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 9.0>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealFirst

func.func @IntegerRealFirst() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 10.2>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealEqual

func.func @IntegerRealEqual() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealSecond

func.func @IntegerRealSecond() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 9.7>
    %result = bmodelica.lt %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerFirst

func.func @RealIntegerFirst() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 9.7>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerEqual

func.func @RealIntegerEqual() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerSecond

func.func @RealIntegerSecond() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.2>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.lt %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}
