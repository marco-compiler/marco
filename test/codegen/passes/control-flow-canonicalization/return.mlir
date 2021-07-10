// RUN: modelica-opt %s --split-input-file --canonicalize-cfg | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %0[] : memref<i1>
// CHECK-NEXT: %1 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %1, %2 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %3 = modelica.constant #modelica.int<5>
// CHECK-NEXT: %4 = modelica.eq %arg0, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT: modelica.if (%4 : !modelica.bool) {
// CHECK-NEXT:      %true = constant true
// CHECK-NEXT:      memref.store %true, %0[] : memref<i1>
// CHECK-NEXT: }
// CHECK-NEXT: %5 = memref.load %0[] : memref<i1>
// CHECK-NEXT: %false_0 = constant false
// CHECK-NEXT: %6 = cmpi eq, %5, %false_0 : i1
// CHECK-NEXT: scf.if %6 {
// CHECK-NEXT:      modelica.member_store %1, %arg0 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:      %7 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:      %8 = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %9 = modelica.mul %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:      modelica.member_store %1, %9 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: }

modelica.function @foo(%arg0 : !modelica.int) -> (!modelica.int) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
    %2 = modelica.constant #modelica.int<5>
    %3 = modelica.eq %arg0, %2 : (!modelica.int, !modelica.int) -> !modelica.bool
    modelica.if (%3 : !modelica.bool) {
        modelica.return
    }
    modelica.member_store %0, %arg0 : !modelica.member<stack, !modelica.int>
    %4 = modelica.member_load %0 : !modelica.int
    %5 = modelica.constant #modelica.int<2>
    %6 = modelica.mul %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.member_store %0, %6 : !modelica.member<stack, !modelica.int>
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %0[] : memref<i1>
// CHECK-NEXT: %1 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %1, %2 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: modelica.while {
// CHECK-NEXT:      %3 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %4 = scf.if %3 -> (i1) {
// CHECK-NEXT:          %false_0 = constant false
// CHECK-NEXT:          scf.yield %false_0 : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:          %6 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %7 = modelica.constant #modelica.int<10>
// CHECK-NEXT:          %8 = modelica.lt %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:          scf.yield %9 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = unrealized_conversion_cast %4 : i1 to !modelica.bool
// CHECK-NEXT:      modelica.condition (%5 : !modelica.bool)
// CHECK-NEXT: } do {
// CHECK-NEXT:      %3 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:      %4 = modelica.constant #modelica.int<5>
// CHECK-NEXT:      %5 = modelica.eq %3, %4 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:      modelica.if (%5 : !modelica.bool) {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %0[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %false_0 = constant false
// CHECK-NEXT:      %7 = cmpi eq, %6, %false_0 : i1
// CHECK-NEXT:      scf.if %7 {
// CHECK-NEXT:          %8 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %9 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          %10 = modelica.add %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %1, %10 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          %11 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %12 = modelica.constant #modelica.int<2>
// CHECK-NEXT:          %13 = modelica.mul %11, %12 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %1, %13 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:      }
// CHECK-NEXT: }

modelica.function @foo(%arg0 : !modelica.int) -> (!modelica.int) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
    modelica.while {
        %2 = modelica.member_load %0 : !modelica.int
        %3 = modelica.constant #modelica.int<10>
        %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%4 : !modelica.bool)
    } do {
        %2 = modelica.member_load %0 : !modelica.int
        %3 = modelica.constant #modelica.int<5>
        %4 = modelica.eq %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.if (%4 : !modelica.bool) {
            modelica.return
        }
        %5 = modelica.member_load %0 : !modelica.int
        %6 = modelica.constant #modelica.int<1>
        %7 = modelica.add %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %7 : !modelica.member<stack, !modelica.int>
        %8 = modelica.member_load %0 : !modelica.int
        %9 = modelica.constant #modelica.int<2>
        %10 = modelica.mul %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %10 : !modelica.member<stack, !modelica.int>
    }
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %0[] : memref<i1>
// CHECK-NEXT: %1 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %1, %2 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: modelica.while {
// CHECK-NEXT:      %3 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %4 = scf.if %3 -> (i1) {
// CHECK-NEXT:          %false_0 = constant false
// CHECK-NEXT:          scf.yield %false_0 : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:          %6 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %7 = modelica.constant #modelica.int<10>
// CHECK-NEXT:          %8 = modelica.lt %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:          scf.yield %9 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = unrealized_conversion_cast %4 : i1 to !modelica.bool
// CHECK-NEXT:      modelica.condition (%5 : !modelica.bool)
// CHECK-NEXT: } do {
// CHECK-NEXT:      modelica.while {
// CHECK-NEXT:          %5 = memref.load %0[] : memref<i1>
// CHECK-NEXT:          %6 = scf.if %5 -> (i1) {
// CHECK-NEXT:              %false_1 = constant false
// CHECK-NEXT:              scf.yield %false_1 : i1
// CHECK-NEXT:          } else {
// CHECK-NEXT:              %8 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:              %9 = modelica.constant #modelica.int<5>
// CHECK-NEXT:              %10 = modelica.lt %8, %9 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:              %11 = unrealized_conversion_cast %10 : !modelica.bool to i1
// CHECK-NEXT:              scf.yield %11 : i1
// CHECK-NEXT:          }
// CHECK-NEXT:          %7 = unrealized_conversion_cast %6 : i1 to !modelica.bool
// CHECK-NEXT:          modelica.condition (%7 : !modelica.bool)
// CHECK-NEXT:      } do {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %0[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %3 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %false_0 = constant false
// CHECK-NEXT:      %4 = cmpi eq, %3, %false_0 : i1
// CHECK-NEXT:      scf.if %4 {
// CHECK-NEXT:          %5 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %6 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          %7 = modelica.add %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %1, %7 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          %8 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %9 = modelica.constant #modelica.int<2>
// CHECK-NEXT:          %10 = modelica.mul %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %1, %10 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:      }
// CHECK-NEXT: }

modelica.function @foo(%arg0 : !modelica.int) -> (!modelica.int) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
    modelica.while {
        %2 = modelica.member_load %0 : !modelica.int
        %3 = modelica.constant #modelica.int<10>
        %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%4 : !modelica.bool)
    } do {
        modelica.while {
            %2 = modelica.member_load %0 : !modelica.int
            %3 = modelica.constant #modelica.int<5>
            %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
            modelica.condition (%4 : !modelica.bool)
        } do {
            modelica.return
        }
        %2 = modelica.member_load %0 : !modelica.int
        %3 = modelica.constant #modelica.int<1>
        %4 = modelica.add %2, %3 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %4 : !modelica.member<stack, !modelica.int>
        %5 = modelica.member_load %0 : !modelica.int
        %6 = modelica.constant #modelica.int<2>
        %7 = modelica.mul %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %7 : !modelica.member<stack, !modelica.int>
    }
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %0[] : memref<i1>
// CHECK-NEXT: %1 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %1, %2 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %3 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT: %4 = modelica.constant #modelica.int<1>
// CHECK-NEXT: modelica.store %3[], %4 : !modelica.array<stack, !modelica.int>
// CHECK-NEXT: modelica.for {
// CHECK-NEXT:      %7 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %8 = scf.if %7 -> (i1) {
// CHECK-NEXT:          %false_1 = constant false
// CHECK-NEXT:          scf.yield %false_1 : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:          %10 = modelica.load %3[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:          %11 = modelica.constant #modelica.int<10>
// CHECK-NEXT:          %12 = modelica.lte %10, %11 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          %13 = unrealized_conversion_cast %12 : !modelica.bool to i1
// CHECK-NEXT:          scf.yield %13 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = unrealized_conversion_cast %8 : i1 to !modelica.bool
// CHECK-NEXT:      modelica.condition (%9 : !modelica.bool)
// CHECK-NEXT: } body {
// CHECK-NEXT:      %7 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:      %8 = modelica.constant #modelica.int<5>
// CHECK-NEXT:      %9 = modelica.lt %7, %8 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:      modelica.if (%9 : !modelica.bool) {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %0[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %10 = memref.load %0[] : memref<i1>
// CHECK-NEXT:      %false_1 = constant false
// CHECK-NEXT:      %11 = cmpi eq, %10, %false_1 : i1
// CHECK-NEXT:      scf.if %11 {
// CHECK-NEXT:          %12 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          modelica.member_store %1, %12 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          %13 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:          %14 = modelica.load %3[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:          %15 = modelica.add %13, %14 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %1, %15 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:      }
// CHECK-NEXT: } step {
// CHECK-NEXT:      %7 = modelica.load %3[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:      %8 = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %9 = modelica.add %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:      modelica.store %3[], %9 : !modelica.array<stack, !modelica.int>
// CHECK-NEXT: }
// CHECK-NEXT: %5 = memref.load %0[] : memref<i1>
// CHECK-NEXT: %false_0 = constant false
// CHECK-NEXT: %6 = cmpi eq, %5, %false_0 : i1
// CHECK-NEXT: scf.if %6 {
// CHECK-NEXT:      %7 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:      %8 = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %9 = modelica.add %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:      modelica.member_store %1, %9 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:      %10 = modelica.member_load %1 : !modelica.int
// CHECK-NEXT:      %11 = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %12 = modelica.mul %10, %11 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:       modelica.member_store %1, %12 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: }

modelica.function @foo(%arg0 : !modelica.int) -> (!modelica.int) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
    %2 = modelica.alloca : !modelica.array<stack, !modelica.int>
    %3 = modelica.constant #modelica.int<1>
    modelica.store %2[], %3 : !modelica.array<stack, !modelica.int>
    modelica.for {
        %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
        %5 = modelica.constant #modelica.int<10>
        %6 = modelica.lte %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%6 : !modelica.bool)
    } body {
        %4 = modelica.member_load %0 : !modelica.int
        %5 = modelica.constant #modelica.int<5>
        %6 = modelica.lt %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.if (%6 : !modelica.bool) {
            modelica.return
        }
        %7 = modelica.constant #modelica.int<1>
        modelica.member_store %0, %7 : !modelica.member<stack, !modelica.int>
        %8 = modelica.member_load %0 : !modelica.int
        %9 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
        %10 = modelica.add %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %10 : !modelica.member<stack, !modelica.int>
    } step {
        %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
        %5 = modelica.constant #modelica.int<1>
        %6 = modelica.add %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %2[], %6 : !modelica.array<stack, !modelica.int>
    }
    %4 = modelica.member_load %0 : !modelica.int
    %5 = modelica.constant #modelica.int<1>
    %6 = modelica.add %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.member_store %0, %6 : !modelica.member<stack, !modelica.int>
    %7 = modelica.member_load %0 : !modelica.int
    %8 = modelica.constant #modelica.int<2>
    %9 = modelica.mul %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.member_store %0, %9 : !modelica.member<stack, !modelica.int>
}
