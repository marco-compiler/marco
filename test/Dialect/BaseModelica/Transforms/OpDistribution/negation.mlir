// RUN: modelica-opt %s --split-input-file --distribute-neg --cse | FileCheck %s

// CHECK-LABEL: @neg
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = bmodelica.neg %[[neg0]]
// CHECK: return %[[res]]

func.func @neg(%arg0: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.neg %arg0 : !bmodelica.real -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = bmodelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = bmodelica.add %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = bmodelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = bmodelica.add_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.add_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = bmodelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = bmodelica.sub %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.sub %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[neg1:.*]] = bmodelica.neg %[[arg1]]
// CHECK: %[[res:.*]] = bmodelica.sub_ew %[[neg0]], %[[neg1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.sub_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = bmodelica.mul %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.mul %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = bmodelica.mul_ew %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = bmodelica.div %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK: %[[neg0:.*]] = bmodelica.neg %[[arg0]]
// CHECK: %[[res:.*]] = bmodelica.div_ew %[[neg0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
    return %1 : !bmodelica.real
}
