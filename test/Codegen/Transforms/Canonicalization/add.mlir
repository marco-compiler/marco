// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Range as first argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica<range !bmodelica.int>, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica<range !bmodelica.int>
// CHECK: %[[result:.*]] = bmodelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !bmodelica<range !bmodelica.int>, %arg1: !bmodelica.int) -> !bmodelica<range !bmodelica.int> {
    %result = bmodelica.add %arg0, %arg1 : (!bmodelica<range !bmodelica.int>, !bmodelica.int) -> !bmodelica<range !bmodelica.int>
    return %result : !bmodelica<range !bmodelica.int>
}

// -----

// Range as argument argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica<range !bmodelica.int>, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica<range !bmodelica.int>
// CHECK: %[[result:.*]] = bmodelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !bmodelica<range !bmodelica.int>, %arg1: !bmodelica.int) -> !bmodelica<range !bmodelica.int> {
    %result = bmodelica.add %arg1, %arg0 : (!bmodelica.int, !bmodelica<range !bmodelica.int>) -> !bmodelica<range !bmodelica.int>
    return %result : !bmodelica<range !bmodelica.int>
}
