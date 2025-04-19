// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.int) -> !bmodelica.real {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.gt %[[arg0]], %[[zero]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.log10 %arg0 : !bmodelica.int -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// COM: Real operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.real) -> !bmodelica.real {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.gt %[[arg0]], %[[zero]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.log10 %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}