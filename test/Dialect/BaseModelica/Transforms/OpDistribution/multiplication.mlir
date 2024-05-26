// RUN: modelica-opt %s --split-input-file --distribute-mul --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = bmodelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.add %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = bmodelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.add_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.add_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = bmodelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.sub %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.sub %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[mul1:.*]] = bmodelica.mul %[[arg1]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.sub_ew %[[mul0]], %[[mul1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.sub_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.mul %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.mul %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.mul_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.div %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[factor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[mul0:.*]] = bmodelica.mul %[[arg0]], %[[factor]]
// CHECK: %[[res:.*]] = bmodelica.div_ew %[[mul0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}
