// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @add_der
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK: %[[DER_Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_z"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.add %[[DER_X]], %[[DER_Y]]
// CHECK: modelica.member_store %[[DER_Z]], %[[RESULT]]

modelica.function @add(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"add_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}

// -----

// CHECK-LABEL: @sub_der
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK: %[[DER_Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_z"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.sub %[[DER_X]], %[[DER_Y]]
// CHECK: modelica.member_store %[[DER_Z]], %[[RESULT]]

modelica.function @sub(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"sub_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.sub %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}

// -----

// CHECK-LABEL: @mul_der
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK: %[[DER_Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_z"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[TMP0:[a-zA-Z0-9]*]] = modelica.mul %[[DER_X]], %[[Y]]
// CHECK: %[[TMP1:[a-zA-Z0-9]*]] = modelica.mul %[[X]], %[[DER_Y]]
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.add %[[TMP0]], %[[TMP1]]
// CHECK: modelica.member_store %[[DER_Z]], %[[RESULT]]

modelica.function @mul(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"mul_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}

// -----

// CHECK-LABEL: @div_der
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_X:[a-zA-Z0-9]*]] : !modelica.real
// CHECK-SAME: %[[DER_Y:[a-zA-Z0-9]*]] : !modelica.real
// CHECK: %[[DER_Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "der_z"} : !modelica.member<stack, !modelica.real>
// CHECK: %[[TMP0:[a-zA-Z0-9]*]] = modelica.mul %[[DER_X]], %[[Y]]
// CHECK: %[[TMP1:[a-zA-Z0-9]*]] = modelica.mul %[[X]], %[[DER_Y]]
// CHECK: %[[NUM:[a-zA-Z0-9]*]] = modelica.sub %[[TMP0]], %[[TMP1]]
// CHECK: %[[TWO:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<2>
// CHECK: %[[DEN:[a-zA-Z0-9]*]] = modelica.pow %[[Y]], %[[TWO]]
// CHECK: %[[RESULT:[a-zA-Z0-9]*]] = modelica.div %[[NUM]], %[[DEN]]
// CHECK: modelica.member_store %[[DER_Z]], %[[RESULT]]

modelica.function @div(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"div_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.div %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
}

