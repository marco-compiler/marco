// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operands

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.int, %[[rhs:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.int, %arg1: !bmodelica.int) -> !bmodelica.real {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: taking the remainder of a division by 0"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.neq %[[rhs]], %[[zero]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.mod %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// COM: Real operands

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.real, %[[rhs:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.real, %arg1: !bmodelica.real) -> !bmodelica.real {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: taking the remainder of a division by 0"} {
    // CHECK-NEXT:      %[[epsilon:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:      %[[rhs_abs:.*]] = bmodelica.abs %[[rhs]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.gte %[[rhs_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.mod %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}