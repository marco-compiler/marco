// RUN: modelica-opt %s --split-input-file --distribute-mul --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @add(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @add_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @sub(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @sub_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func @mul(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func @mul_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func @div(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func @div_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}
