// RUN: modelica-opt %s --split-input-file --scalarize-functions | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.ptr<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.ptr<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.sin %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @foo(%arg0 : !modelica.ptr<?x!modelica.real>) -> (!modelica.ptr<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.sin %arg0 : !modelica.ptr<?x!modelica.real> -> !modelica.ptr<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.ptr<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.ptr<heap, ?x!modelica.real>
}