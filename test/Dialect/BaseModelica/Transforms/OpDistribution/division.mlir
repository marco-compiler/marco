// RUN: modelica-opt %s --split-input-file --distribute-div --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.add %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.add_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.add_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.sub %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.sub %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.sub_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.sub_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.mul %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.mul %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.mul_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.div %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.div_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    %1 = bmodelica.constant #bmodelica<int 2>
    %2 = bmodelica.div %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    return %2 : !bmodelica.int
}
