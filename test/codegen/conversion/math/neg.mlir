// RUN: modelica-opt %s --convert-functions --convert-modelica --convert-to-cfg --convert-to-llvm | FileCheck %s

// CHECK-LABEL: @integerScalar
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: i64
// CHECK-NEXT: %[[ZERO:[a-zA-Z0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %{{.*}} = llvm.sub %[[ZERO]], %[[ARG0]] : i64
// CHECK-NEXT: llvm.return

modelica.function @integerScalar(%arg0 : !modelica.int) -> () attributes {args_names = ["x"], results_names = []} {
    %0 = modelica.neg %arg0 : !modelica.int -> !modelica.int
    modelica.return
}

// CHECK-LABEL: @floatScalar
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: f64
// CHECK-NEXT: %{{.*}} = llvm.fneg %[[ARG0]] : f64
// CHECK-NEXT: llvm.return

modelica.function @floatScalar(%arg0 : !modelica.real) -> () attributes {args_names = ["x"], results_names = []} {
    %0 = modelica.neg %arg0 : !modelica.real -> !modelica.real
    modelica.return
}
