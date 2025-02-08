// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @negative

func.func @negative() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -3.14>
    %result = bmodelica.ceil %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real -3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @positive

func.func @positive() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.14>
    %result = bmodelica.ceil %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 4.000000e+00>
    // CHECK: return %[[cst]]
}
