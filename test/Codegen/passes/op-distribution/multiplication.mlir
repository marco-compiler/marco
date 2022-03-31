// RUN: modelica-opt %s --distribute-mul | FileCheck %s

// CHECK-LABEL: @constant
// CHECK-NEXT: %[[cst:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul %[[cst]], %[[factor]]
// CHECK: return %[[res]]

func @constant() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// CHECK-LABEL: @add
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[cst1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @add() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.add %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @add_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[cst1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @add_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.add_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @sub
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[cst1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @sub() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.sub %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @sub_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[mul1:[a-zA-Z0-9]*]] = modelica.mul %[[cst1]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func @sub_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.sub_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @mul
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul %[[mul0]], %[[cst1]]
// CHECK: return %[[res]]

func @mul() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @mul_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul_ew %[[mul0]], %[[cst1]]
// CHECK: return %[[res]]

func @mul_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.mul_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @div
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div %[[mul0]], %[[cst1]]
// CHECK: return %[[res]]

func @div() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}

// CHECK-LABEL: @div_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT: %[[factor:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:[a-zA-Z0-9]*]] = modelica.mul %[[cst0]], %[[factor]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div_ew %[[mul0]], %[[cst1]]
// CHECK: return %[[res]]

func @div_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.div_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.constant #modelica.int<2>
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
    return %4 : !modelica.int
}
