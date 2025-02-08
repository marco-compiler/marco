// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @negativeInteger

func.func @negativeInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int -2>
    %result = bmodelica.abs %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @zeroInteger

func.func @zeroInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.abs %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @positiveInteger

func.func @positiveInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 2>
    %result = bmodelica.abs %x : !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @negativeReal

func.func @negativeReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real -1.5>
    %result = bmodelica.abs %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real 1.500000e+00>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @zeroReal

func.func @zeroReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.abs %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @positiveReal

func.func @positiveReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 1.5>
    %result = bmodelica.abs %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<real 1.500000e+00>
    // CHECK-NEXT: return %[[value]]
}
