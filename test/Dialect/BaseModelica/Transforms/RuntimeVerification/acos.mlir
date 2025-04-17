// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.int) -> !bmodelica.real {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[lower_bound:.*]] = bmodelica.constant #bmodelica<int -1> : !bmodelica.int
    // CHECK-NEXT:  %[[upper_bound:.*]] = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg0]], %[[lower_bound]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lte %[[arg0]], %[[upper_bound]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.acos %arg0 : !bmodelica.int -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// COM: Real operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.real) -> !bmodelica.real {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[lower_bound:.*]] = bmodelica.constant #bmodelica<real -1.000000e+00> : !bmodelica.real
    // CHECK-NEXT:  %[[upper_bound:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00> : !bmodelica.real
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg0]], %[[lower_bound]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lte %[[arg0]], %[[upper_bound]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.acos %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}