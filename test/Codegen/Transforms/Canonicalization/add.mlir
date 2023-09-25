// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Range as first argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica<range !modelica.int>, %[[arg1:.*]]: !modelica.int) -> !modelica<range !modelica.int>
// CHECK: %[[result:.*]] = modelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !modelica<range !modelica.int>, %arg1: !modelica.int) -> !modelica<range !modelica.int> {
    %result = modelica.add %arg0, %arg1 : (!modelica<range !modelica.int>, !modelica.int) -> !modelica<range !modelica.int>
    return %result : !modelica<range !modelica.int>
}

// -----

// Range as argument argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica<range !modelica.int>, %[[arg1:.*]]: !modelica.int) -> !modelica<range !modelica.int>
// CHECK: %[[result:.*]] = modelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !modelica<range !modelica.int>, %arg1: !modelica.int) -> !modelica<range !modelica.int> {
    %result = modelica.add %arg1, %arg0 : (!modelica.int, !modelica<range !modelica.int>) -> !modelica<range !modelica.int>
    return %result : !modelica<range !modelica.int>
}
