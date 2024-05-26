// RUN: modelica-opt %s --split-input-file --inline | FileCheck %s

// Inline attribute set to 'true'

bmodelica.raw_function @foo(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int attributes {inline = true} {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    bmodelica.raw_return %0 : !bmodelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK:       %[[result:.*]] = bmodelica.add %[[arg0]], %[[arg1]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.call @foo(%arg0, %arg1) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %0 : !bmodelica.int
}

// -----

// Inline attribute set to 'false'

bmodelica.raw_function @foo(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int attributes {inline = false} {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    bmodelica.raw_return %0 : !bmodelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK:       %[[result:.*]] = bmodelica.call @foo(%[[arg0]], %[[arg1]]) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.call @foo(%arg0, %arg1) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %0 : !bmodelica.int
}

// -----

// Missing inline attribute

bmodelica.raw_function @foo(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    bmodelica.raw_return %0 : !bmodelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK:       %[[result:.*]] = bmodelica.call @foo(%[[arg0]], %[[arg1]]) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.call @foo(%arg0, %arg1) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %0 : !bmodelica.int
}
