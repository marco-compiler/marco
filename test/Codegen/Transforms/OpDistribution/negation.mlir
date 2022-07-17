// RUN: modelica-opt %s --split-input-file --distribute-neg --cse | FileCheck %s

// CHECK-LABEL: @neg
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = modelica.neg %[[neg0]]
// CHECK: return %[[res]]

func.func @neg(%arg0: !modelica.int) -> !modelica.int {
    %0 = modelica.neg %arg0 : !modelica.int -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = modelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = modelica.add %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = modelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = modelica.add_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.add_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = modelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = modelica.sub %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = modelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = modelica.sub_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.sub_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = modelica.mul %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = modelica.mul_ew %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.mul_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = modelica.div %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.int
// CHECK: %[[neg0:.*]] = modelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = modelica.div_ew %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !modelica.int, %arg1: !modelica.int) -> !modelica.int {
    %0 = modelica.div_ew %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.neg %0 : !modelica.int -> !modelica.int
    return %1 : !modelica.int
}
