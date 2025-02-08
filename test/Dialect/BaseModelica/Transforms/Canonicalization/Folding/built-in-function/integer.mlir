// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @negative

func.func @negative() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<real -3.14>
    %result = bmodelica.integer %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<int -4>
    // CHECK-NEXT: return %[[value]]
}

// -----

// CHECK-LABEL: @positive

func.func @positive() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<real 3.14>
    %result = bmodelica.integer %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK-NEXT: %[[value:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK-NEXT: return %[[value]]
}
