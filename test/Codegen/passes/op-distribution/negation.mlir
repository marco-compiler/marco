// RUN: modelica-opt %s --distribute-neg | FileCheck %s

// CHECK-LABEL: @constant
// CHECK-NEXT: %[[cst:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.neg %[[cst]]
// CHECK: return %[[res]]

func @constant() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// CHECK-LABEL: @neg
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.neg %[[neg0]]
// CHECK: return %[[res]]

func @neg() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    %2 = modelica.neg %1 : !modelica.int -> !modelica.int
    return %2 : !modelica.int
}

// CHECK-LABEL: @add
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[neg1:[a-zA-Z0-9]*]] = modelica.neg %[[cst1]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func @add() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.add %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @add_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[neg1:[a-zA-Z0-9]*]] = modelica.neg %[[cst1]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.add_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func @add_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.add_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @sub
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[neg1:[a-zA-Z0-9]*]] = modelica.neg %[[cst1]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func @sub() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.sub %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @sub_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[neg1:[a-zA-Z0-9]*]] = modelica.neg %[[cst1]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.sub_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func @sub_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.sub_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @mul
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul %[[neg0]], %[[cst1]]
// CHECK: return %[[res]]

func @mul() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @mul_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.mul_ew %[[neg0]], %[[cst1]]
// CHECK: return %[[res]]

func @mul_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.mul_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @div
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div %[[neg0]], %[[cst1]]
// CHECK: return %[[res]]

func @div() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.div %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}

// CHECK-LABEL: @div_ew
// CHECK-NEXT: %[[cst0:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT: %[[cst1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1>
// CHECK: %[[neg0:[a-zA-Z0-9]*]] = modelica.neg %[[cst0]]
// CHECK: %[[res:[a-zA-Z0-9]*]] = modelica.div_ew %[[neg0]], %[[cst1]]
// CHECK: return %[[res]]

func @div_ew() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    %1 = modelica.constant #modelica.int<1>
    %2 = modelica.div_ew %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.neg %2 : !modelica.int -> !modelica.int
    return %3 : !modelica.int
}
