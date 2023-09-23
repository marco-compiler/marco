// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// Iterable as first argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica<iterable !modelica.int>, %[[arg1:.*]]: !modelica.int) -> !modelica<iterable !modelica.int>
// CHECK: %[[result:.*]] = modelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !modelica<iterable !modelica.int>, %arg1: !modelica.int) -> !modelica<iterable !modelica.int> {
    %result = modelica.add %arg0, %arg1 : (!modelica<iterable !modelica.int>, !modelica.int) -> !modelica<iterable !modelica.int>
    return %result : !modelica<iterable !modelica.int>
}

// -----

// Iterable as argument argument.

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg0:.*]]: !modelica<iterable !modelica.int>, %[[arg1:.*]]: !modelica.int) -> !modelica<iterable !modelica.int>
// CHECK: %[[result:.*]] = modelica.add %arg0, %arg1
// CHECK-NEXT: return %[[result]]

func.func @test(%arg0: !modelica<iterable !modelica.int>, %arg1: !modelica.int) -> !modelica<iterable !modelica.int> {
    %result = modelica.add %arg1, %arg0 : (!modelica.int, !modelica<iterable !modelica.int>) -> !modelica<iterable !modelica.int>
    return %result : !modelica<iterable !modelica.int>
}
