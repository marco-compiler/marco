// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @foo1
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<2x!modelica.real>
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: %[[Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.array<2x!modelica.real>
// CHECK-SAME: %[[DER_Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: -> (!modelica.real)
// CHECK-SAME: args_names = ["x", "y", "z", "der_x", "der_z"]
// CHECK-SAME: results_names = ["der_t"]
// CHECK: %[[T:[a-zA-Z0-9]*]] = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[DER_T:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.member_load %[[DER_T]] : !modelica.real
// CHECK: modelica.return %[[RESULT]]

modelica.function @foo(%arg0 : !modelica.array<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z"], results_names = ["t"], derivative = #modelica.derivative<"foo1", 1>} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.member_load %0 : !modelica.real
    modelica.return %1 : !modelica.real
}

// -----

// CHECK-LABEL: @foo2
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<2x!modelica.real>
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: %[[Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.array<2x!modelica.real>
// CHECK-SAME: %[[DER_Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_2_X:[a-zA-Z0-9]*]] : !modelica.array<2x!modelica.real>
// CHECK-SAME: %[[DER_2_Z:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: -> (!modelica.real)
// CHECK-SAME: args_names = ["x", "y", "z", "der_x", "der_z", "der_2_x", "der_2_z"]
// CHECK-SAME: results_names = ["der_2_t"]
// CHECK: %[[T:[a-zA-Z0-9]*]] = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[DER_T:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[DER_2_T:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_2_t"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.member_load %[[DER_2_T]] : !modelica.real
// CHECK: modelica.return %[[RESULT]]

modelica.function @foo1(%arg0 : !modelica.array<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real, %arg3 : !modelica.array<2x!modelica.real>, %arg4 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z", "der_x", "der_z"], results_names = ["der_t"], derivative = #modelica.derivative<"foo2", 2>} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.member_create {name = "der_t"} : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %1 : !modelica.real
    %3 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}