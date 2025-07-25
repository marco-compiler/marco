// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK-SAME: (%[[arg:.*]]: !bmodelica.int)

func.func @Integer(%0: !bmodelica.int) -> (!bmodelica.int) {
    %cst = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.add %0, %cst : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %result : !bmodelica.int

    // CHECK: return %[[arg]]
}

// -----

// CHECK-LABEL: @Real
// CHECK-SAME: (%[[arg:.*]]: !bmodelica.real)

func.func @Real(%0: !bmodelica.real) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.add %0, %cst : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: return %[[arg]]
}

// -----

// CHECK-LABEL: @IntegerReal
// CHECK-SAME: (%[[arg:.*]]: !bmodelica.int)

func.func @IntegerReal(%0: !bmodelica.int) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<real 0.0>
    %result = bmodelica.add %0, %cst : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: %[[result:.*]] = bmodelica.cast %[[arg]] : !bmodelica.int -> !bmodelica.real
    // CHECK: return %[[result]]
}

// -----

// CHECK-LABEL: @RealInteger
// CHECK-SAME: (%[[arg:.*]]: !bmodelica.real)

func.func @RealInteger(%0: !bmodelica.real) -> (!bmodelica.real) {
    %cst = bmodelica.constant #bmodelica<int 0>
    %result = bmodelica.add %0, %cst : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    return %result : !bmodelica.real

    // CHECK: return %[[arg]]
}
