// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @rangeScalar
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica<range !bmodelica.int>, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica<range !bmodelica.int>

func.func @rangeScalar(%arg0: !bmodelica<range !bmodelica.int>, %arg1: !bmodelica.int) -> !bmodelica<range !bmodelica.int> {
    %result = bmodelica.add %arg0, %arg1 : (!bmodelica<range !bmodelica.int>, !bmodelica.int) -> !bmodelica<range !bmodelica.int>
    return %result : !bmodelica<range !bmodelica.int>

    // CHECK: %[[result:.*]] = bmodelica.add %arg0, %arg1
    // CHECK-NEXT: return %[[result]]
}

// -----

// CHECK-LABEL: @scalarRange
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica<range !bmodelica.int>, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica<range !bmodelica.int>

func.func @scalarRange(%arg0: !bmodelica<range !bmodelica.int>, %arg1: !bmodelica.int) -> !bmodelica<range !bmodelica.int> {
    %result = bmodelica.add %arg1, %arg0 : (!bmodelica.int, !bmodelica<range !bmodelica.int>) -> !bmodelica<range !bmodelica.int>
    return %result : !bmodelica<range !bmodelica.int>

    // CHECK: %[[result:.*]] = bmodelica.add %arg0, %arg1
    // CHECK-NEXT: return %[[result]]
}
