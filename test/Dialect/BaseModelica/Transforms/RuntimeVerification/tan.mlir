// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer operand

// CHECK:      %[[arg:.*]] = bmodelica.constant #bmodelica<int {{-?[[:digit:]]+}}> : !bmodelica.int
// CHECK-NEXT: %{{[[:digit:]]+}} = bmodelica.tan %[[arg]] : !bmodelica.int -> !bmodelica.real

%0 = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
%1 = bmodelica.tan %0 : !bmodelica.int -> !bmodelica.real

// -----

// COM: Real operand

// CHECK:      %[[arg:.*]] = bmodelica.constant #bmodelica<real {{(-?[[:digit:]]+[.][[:digit:]]{6}e[+-][[:digit:]]{2,})}}> : !bmodelica.real
// CHECK-NEXT: bmodelica.assert {level = 2 : i64, message = "Model error: Argument of tan is invalid. It should not be a multiple of pi/2"} {
// CHECK-NEXT:   %[[arg_abs:.*]] = bmodelica.abs %0 : !bmodelica.real -> !bmodelica.real
// CHECK-NEXT:   %[[pi:.*]] = bmodelica.constant #bmodelica<real 3.1415926535897931> : !bmodelica.real
// CHECK-NEXT:   %[[pi2:.*]] = bmodelica.constant #bmodelica<real 1.5707963267948966> : !bmodelica.real
// CHECK-NEXT:   %[[epsilon:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
// CHECK-NEXT:   %[[arg_modpi2:.*]] = bmodelica.mod %[[arg_abs]], %[[pi2]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:   %[[arg_modpi:.*]] = bmodelica.mod %[[arg_abs]], %[[pi]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:   %[[is_mulpi_hi:.*]] = bmodelica.lte %7, %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
// CHECK-NEXT:   %[[diff:.*]] = bmodelica.sub %[[arg_modpi]], %[[pi]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:   %[[diff_abs:.*]] = bmodelica.abs %[[diff]] : !bmodelica.real -> !bmodelica.real
// CHECK-NEXT:   %[[is_mulpi_lo:.*]] = bmodelica.lte %[[diff_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
// CHECK-NEXT:   %[[is_mulpi:.*]] = bmodelica.or %[[is_mulpi_lo]], %[[is_mulpi_hi]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
// CHECK-NEXT:   %[[isnot_mulpi2_hi:.*]] = bmodelica.gte %[[arg_modpi2]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
// CHECK-NEXT:   %[[diff:.*]] = bmodelica.sub %[[arg_modpi2]], %[[pi2]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:   %[[diff_abs:.*]] = bmodelica.abs %[[diff]] : !bmodelica.real -> !bmodelica.real
// CHECK-NEXT:   %[[isnot_mulpi2_lo:.*]] = bmodelica.gte %[[diff_abs]], %[[epsilon]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
// CHECK-NEXT:   %[[isnot_mulpi2:.*]] = bmodelica.and %[[isnot_mulpi2_lo]], %[[isnot_mulpi2_hi]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
// CHECK-NEXT:   %[[cond:.*]] = bmodelica.or %[[is_mulpi]], %[[isnot_mulpi2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
// CHECK-NEXT:   bmodelica.yield %[[cond]] : !bmodelica.bool
// CHECK-NEXT: }
// CHECK-NEXT: %{{[[:digit:]]+}} = bmodelica.tan %[[arg]] : !bmodelica.real -> !bmodelica.real

%0 = bmodelica.constant #bmodelica<real 1.000000e+00> : !bmodelica.real
%1 = bmodelica.tan %0 : !bmodelica.real -> !bmodelica.real
