// RUN: modelica-opt %s --split-input-file --scalarize-functions | FileCheck %s

// CHECK-LABEL: @abs
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.abs %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @abs(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.abs %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @acos
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.acos %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @acos(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.acos %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @asin
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.asin %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @asin(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.asin %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @atan
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.atan %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @atan(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.atan %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @atan2
// CHECK-SAME: %[[Y:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Z:[a-zA-Z0-9]*]] = modelica.member_create {name = "z"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[Y_SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[Y]][%[[INDEX]]]
// CHECK-NEXT: %[[Y_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[Y_SUBSCRIPTION]][]
// CHECK-NEXT: %[[X_SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[X_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[X_SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.atan2 %[[Y_LOAD]], %[[X_LOAD]] : (!modelica.real, !modelica.real) -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Z]], %[[ARRAY]]

modelica.function @atan2(%arg0 : !modelica.array<?x!modelica.real>, %arg1 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x", "y"], results_names = ["z"]} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.atan2 %arg0, %arg1 : (!modelica.array<?x!modelica.real>, !modelica.array<?x!modelica.real>) -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @cos
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.cos %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @cos(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.cos %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @cosh
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.cosh %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @cosh(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.cosh %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @exp
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.exp %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @exp(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.exp %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @log
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.log %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @log(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.log %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @log10
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.log10 %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @log10(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.log10 %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @sign
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.sign %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @sign(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.sign %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @sin
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.sin %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @sin(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.sin %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @sinh
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.sinh %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @sinh(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.sinh %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @sqrt
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.sqrt %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @sqrt(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.sqrt %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @tan
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.tan %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @tan(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.tan %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}

// -----

// CHECK-LABEL: @tanh
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.array<?x!modelica.real>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
// CHECK: %[[ARRAY:[a-zA-Z0-9]*]] = modelica.alloc %{{.*}} : index -> !modelica.array<heap, ?x!modelica.real>
// CHECK: scf.for %[[INDEX:[a-zA-Z0-9]*]]
// CHECK-NEXT: %[[SUBSCRIPTION:[a-zA-Z0-9]*]] = modelica.subscription %[[X]][%[[INDEX]]]
// CHECK-NEXT: %[[LOAD:[a-zA-Z0-9]*]] = modelica.load %[[SUBSCRIPTION]][]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9]*]] = modelica.tanh %[[LOAD]] : !modelica.real -> !modelica.real
// CHECK-NEXT: %[[DESTINATION:[a-zA-Z0-9]*]] = modelica.subscription %[[ARRAY]]
// CHECK-NEXT: modelica.assignment %[[SCALAR]], %[[DESTINATION]]
// CHECK-NEXT: }
// CHECK: modelica.member_store %[[Y]], %[[ARRAY]]

modelica.function @tanh(%arg0 : !modelica.array<?x!modelica.real>) -> (!modelica.array<heap, ?x!modelica.real>) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<heap, ?x!modelica.real>
    %1 = modelica.tanh %arg0 : !modelica.array<?x!modelica.real> -> !modelica.array<heap, ?x!modelica.real>
    modelica.member_store %0, %1 : !modelica.member<heap, ?x!modelica.real>
    %2 = modelica.member_load %0 : !modelica.array<heap, ?x!modelica.real>
    modelica.return %2 : !modelica.array<heap, ?x!modelica.real>
}
