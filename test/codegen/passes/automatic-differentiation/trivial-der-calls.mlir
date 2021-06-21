// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @sin
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[COS:[a-zA-Z0-9]*]] = modelica.cos %[[X]] : !modelica.real -> !modelica.real
// CHECK: %[[MUL:[a-zA-Z0-9]*]] = modelica.mul_ew %[[COS]], %[[DER_X]] : (!modelica.real, !modelica.real) -> !modelica.real
// CHECK: modelica.member_store %[[Y]], %[[MUL]]

modelica.function @sin(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "der_x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.sin %arg0 : !modelica.real -> !modelica.real
    %2 = modelica.der %1 : !modelica.real -> !modelica.real
    modelica.member_store %0, %2 : !modelica.member<stack, !modelica.real>
    %3 = modelica.member_load %0 : !modelica.real
    modelica.return %3 : !modelica.real
}
