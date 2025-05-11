// RUN: modelica-opt %s --split-input-file --distribute-div --canonicalize | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.add %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @add(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.add %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @add_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.add_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @add_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.add_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.sub %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @sub(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.sub %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @sub_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[div1:.*]] = bmodelica.div %[[arg1]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.sub_ew %[[div0]], %[[div1]]
// CHECK: return %[[res]]

func.func @sub_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.sub_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mul
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.mul %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.mul %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @mul_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.mul_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @mul_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.mul_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @div
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.div %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}

// -----

// CHECK-LABEL: @div_ew
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT: %[[divisor:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK: %[[div0:.*]] = bmodelica.div %[[arg0]], %[[divisor]]
// CHECK: %[[res:.*]] = bmodelica.div_ew %[[div0]], %[[arg1]]
// CHECK: return %[[res]]

func.func @div_ew(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    %1 = bmodelica.constant #bmodelica<real 2.0>
    %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    return %2 : !bmodelica.real
}
