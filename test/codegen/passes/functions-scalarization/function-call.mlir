// RUN: modelica-opt %s --split-input-file --scalarize-functions  | FileCheck %s

modelica.function @callee(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    modelica.member_store %0, %arg0 : !modelica.member<stack, !modelica.real>
    %1 = modelica.member_load %0 : !modelica.real
    modelica.return %1 : !modelica.real
}

// CHECK-LABEL: @caller
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.call @callee(%[[LOAD]]) : (!modelica.real) -> (!modelica.real)
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @caller(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.call @callee(%arg0) : (!modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>)
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}
