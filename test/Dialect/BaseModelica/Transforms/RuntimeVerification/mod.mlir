// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operands

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.int, %[[rhs:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.real {

    // CHECK:       bmodelica.assert {
    // CHECK-NEXT:  %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.neq %[[rhs]], %[[zero]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.mod %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// COM: Real operands

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.real, %[[rhs:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[epsilon:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:  %[[rhs_abs:.*]] = bmodelica.abs %[[rhs]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.gte %[[rhs_abs]], %[[epsilon]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.mod %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}