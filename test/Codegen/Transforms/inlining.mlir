// RUN: modelica-opt %s --split-input-file --inline | FileCheck %s

// Inline attribute set to 'true'

modelica.raw_function @foo(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int attributes {inline = true} {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.raw_return %0 : !modelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK:       %[[result:.*]] = modelica.add %[[arg0]], %[[arg1]] : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.call @foo(%arg0, %arg1) : (!modelica.int, !modelica.int) -> !modelica.int
    return %0 : !modelica.int
}

// -----

// Inline attribute set to 'false'

modelica.raw_function @foo(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int attributes {inline = false} {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.raw_return %0 : !modelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK:       %[[result:.*]] = modelica.call @foo(%[[arg0]], %[[arg1]]) : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.call @foo(%arg0, %arg1) : (!modelica.int, !modelica.int) -> !modelica.int
    return %0 : !modelica.int
}

// -----

// Missing inline attribute

modelica.raw_function @foo(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.raw_return %0 : !modelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK:       %[[result:.*]] = modelica.call @foo(%[[arg0]], %[[arg1]]) : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK:       return %[[result]]

func.func @test(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.call @foo(%arg0, %arg1) : (!modelica.int, !modelica.int) -> !modelica.int
    return %0 : !modelica.int
}
