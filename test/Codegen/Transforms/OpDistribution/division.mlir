// RUN: modelica-opt %s --split-input-file --distribute-div --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:[a-zA-Z0-9]*]] = modelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func @add(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:[a-zA-Z0-9]*]] = modelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func @add_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:[a-zA-Z0-9]*]] = modelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func @sub(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:[a-zA-Z0-9]*]] = modelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func @sub_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func @mul(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func @mul_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func @div(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:[a-zA-Z0-9]*]]: !modelica.int, %[[arg1:[a-zA-Z0-9]*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[divisor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[div0:[a-zA-Z0-9]*]] = modelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func @div_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}
