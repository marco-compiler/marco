// RUN: modelica-opt %s --split-input-file --distribute-mul --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.add %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.add_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.sub %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = modelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.sub_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.mul %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.mul_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.div %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK-NEXT: %[[factor:.*]] = modelica.constant #modelica.int<2>
// CHECK: %[[mul0:.*]] = modelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = modelica.div_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.constant #modelica.int<2>
    %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
    return %2 : !modelica.int
}
