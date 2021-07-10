// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @bar
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: -> (!modelica.real)
// CHECK-SAME: args_names = ["x", "y"]
// CHECK-SAME: results_names = ["z"]
// CHECK: %[[PDER_X_X:[a-zA-Z0-9]*]] = modelica.member_create {name = "pder_x_x"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[PDER_X_X_SEED:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<1.000000>
// CHECK: modelica.member_store %[[PDER_X_X]], %[[PDER_X_X_SEED]]
// CHECK: %[[PDER_X_Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "pder_x_y"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[PDER_X_Y_SEED:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<0.000000>
// CHECK: modelica.member_store %[[PDER_X_Y]], %[[PDER_X_Y_SEED]]
// CHECK: %[[PDER_X_Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "pder_x_z"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[PDER_X_Z_SEED:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<0.000000>
// CHECK: modelica.member_store %[[PDER_X_Z]], %[[PDER_X_Z_SEED]]

modelica.function @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"]} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
}

modelica.der_function @bar {derived_function = "foo", independent_vars = ["x"]}