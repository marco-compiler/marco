// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @FalseFalse

func.func @FalseFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool false>
    %y = bmodelica.constant #bmodelica<bool false>
    %result = bmodelica.or %x, %y : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @FalseTrue

func.func @FalseTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool false>
    %y = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.or %x, %y : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @TrueFalse

func.func @TrueFalse() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool true>
    %y = bmodelica.constant #bmodelica<bool false>
    %result = bmodelica.or %x, %y : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @TrueTrue

func.func @TrueTrue() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool true>
    %y = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.or %x, %y : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}
