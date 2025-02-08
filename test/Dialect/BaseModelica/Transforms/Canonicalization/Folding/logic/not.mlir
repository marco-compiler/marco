// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @False

func.func @False() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool false>
    %result = bmodelica.not %x : !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @True

func.func @True() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.not %x : !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool false>
    // CHECK: return %[[cst]]
}
