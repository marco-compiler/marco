// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @negative

func.func @negative() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -2.5>
    %result = bmodelica.sign %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real -1.000000e+00>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @zero

func.func @zero() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.sign %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @positive

func.func @positive() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 2.5>
    %result = bmodelica.sign %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK-NEXT: return %[[value]]
}
