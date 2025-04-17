// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.int) -> !bmodelica.real {

    // CHECK: %{{.*}} = bmodelica.tan %[[arg0]] : !bmodelica.int -> !bmodelica.real

    %0 = bmodelica.tan %arg0 : !bmodelica.int -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// COM: Real operand

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.real) -> !bmodelica.real {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: Argument of tan is invalid. It should not be a multiple of pi/2"} {
    // CHECK-NEXT:      %[[arg_abs:.*]] = bmodelica.abs %[[arg0]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[pi:.*]] = bmodelica.constant #bmodelica<real 3.1415926535897931> : !bmodelica.real
    // CHECK-NEXT:      %[[pi2:.*]] = bmodelica.constant #bmodelica<real 1.5707963267948966> : !bmodelica.real
    // CHECK-NEXT:      %[[epsilon:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:      %[[arg_modpi2:.*]] = bmodelica.mod %[[arg_abs]], %[[pi2]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    // CHECK-NEXT:      %[[arg_modpi:.*]] = bmodelica.mod %[[arg_abs]], %[[pi]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    // CHECK-NEXT:      %[[is_mulpi_hi:.*]] = bmodelica.lte %[[arg_modpi]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[diff:.*]] = bmodelica.sub %[[arg_modpi]], %[[pi]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    // CHECK-NEXT:      %[[diff_abs:.*]] = bmodelica.abs %[[diff]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[is_mulpi_lo:.*]] = bmodelica.lte %[[diff_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[is_mulpi:.*]] = bmodelica.or %[[is_mulpi_lo]], %[[is_mulpi_hi]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      %[[isnot_mulpi2_hi:.*]] = bmodelica.gte %[[arg_modpi2]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[diff:.*]] = bmodelica.sub %[[arg_modpi2]], %[[pi2]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    // CHECK-NEXT:      %[[diff_abs:.*]] = bmodelica.abs %[[diff]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[isnot_mulpi2_lo:.*]] = bmodelica.gte %[[diff_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[isnot_mulpi2:.*]] = bmodelica.and %[[isnot_mulpi2_lo]], %[[isnot_mulpi2_hi]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.or %[[is_mulpi]], %[[isnot_mulpi2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.tan %arg0 : !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}
