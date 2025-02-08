// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @IntegerTrue

func.func @IntegerTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 9>
    %result = bmodelica.neq %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerFalse

func.func @IntegerFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.neq %x, %y : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealTrue

func.func @RealTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 9.0>
    %result = bmodelica.neq %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealFalse

func.func @RealFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.neq %x, %y : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealTrue

func.func @IntegerRealTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 9.7>
    %result = bmodelica.neq %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerRealFalse

func.func @IntegerRealFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 10>
    %y = bmodelica.constant #bmodelica<real 10.0>
    %result = bmodelica.neq %x, %y : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerTrue

func.func @RealIntegerTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 9.7>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.neq %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealIntegerFalse

func.func @RealIntegerFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 10.0>
    %y = bmodelica.constant #bmodelica<int 10>
    %result = bmodelica.neq %x, %y : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}
