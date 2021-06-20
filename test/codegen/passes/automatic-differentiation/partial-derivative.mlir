// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @foo1
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.ptr<2x!modelica.real>
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: %[[Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.ptr<2x!modelica.real>
// CHECK-SAME: %[[DER_Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: -> (!modelica.real)
// CHECK-SAME: args_names = ["x", "y", "z", "der_x", "der_z"]
// CHECK-SAME: results_names = ["der_t"]
// CHECK: %[[T:[a-zA-Z0-9]*]] = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[DER_T:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.member_load %[[DER_T]] : !modelica.real
// CHECK: modelica.return %[[RESULT]]

modelica.function @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"]} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}

modelica.der_function @bar {derived_function = "foo", independent_vars = ["x"]}