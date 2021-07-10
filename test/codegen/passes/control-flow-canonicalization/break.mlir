// RUN: modelica-opt %s --split-input-file --canonicalize-cfg | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %2[] : memref<i1>
// CHECK-NEXT: modelica.while {
// CHECK-NEXT:      %3 = memref.load %2[] : memref<i1>
// CHECK-NEXT:      %4 = scf.if %3 -> (i1) {
// CHECK-NEXT:          %false_0 = constant false
// CHECK-NEXT:          scf.yield %false_0 : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:          %6 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %7 = modelica.constant #modelica.int<10>
// CHECK-NEXT:          %8 = modelica.lt %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:          scf.yield %9 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = unrealized_conversion_cast %4 : i1 to !modelica.bool
// CHECK-NEXT:      modelica.condition (%5 : !modelica.bool)
// CHECK-NEXT:  } do {
// CHECK-NEXT:      %3 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:      %4 = modelica.constant #modelica.int<5>
// CHECK-NEXT:      %5 = modelica.eq %3, %4 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:      modelica.if (%5 : !modelica.bool) {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %2[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = memref.load %2[] : memref<i1>
// CHECK-NEXT:      %false_0 = constant false
// CHECK-NEXT:      %7 = cmpi eq, %6, %false_0 : i1
// CHECK-NEXT:      scf.if %7 {
// CHECK-NEXT:          %8 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %9 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          %10 = modelica.add %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %0, %10 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          %11 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %12 = modelica.constant #modelica.int<2>
// CHECK-NEXT:          %13 = modelica.add %11, %12 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %0, %13 : !modelica.member<stack, !modelica.int>
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
            modelica.break
        }
        %5 = modelica.member_load %0 : !modelica.int
        %6 = modelica.constant #modelica.int<1>
        %7 = modelica.add %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %7 : !modelica.member<stack, !modelica.int>
        %8 = modelica.member_load %0 : !modelica.int
        %9 = modelica.constant #modelica.int<2>
        %10 = modelica.add %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %10 : !modelica.member<stack, !modelica.int>
    }
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %2 = memref.alloca() : memref<i1>
// CHECK-NEXT: %false = constant false
// CHECK-NEXT: memref.store %false, %2[] : memref<i1>
// CHECK-NEXT: modelica.while {
// CHECK-NEXT:      %3 = memref.load %2[] : memref<i1>
// CHECK-NEXT:      %4 = scf.if %3 -> (i1) {
// CHECK-NEXT:          %false_0 = constant false
// CHECK-NEXT:          scf.yield %false_0 : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:          %6 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %7 = modelica.constant #modelica.int<10>
// CHECK-NEXT:          %8 = modelica.lt %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:          scf.yield %9 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = unrealized_conversion_cast %4 : i1 to !modelica.bool
// CHECK-NEXT:      modelica.condition (%5 : !modelica.bool)
// CHECK-NEXT: } do {
// CHECK-NEXT:      %3 = memref.alloca() : memref<i1>
// CHECK-NEXT:      %false_0 = constant false
// CHECK-NEXT:      memref.store %false_0, %3[] : memref<i1>
// CHECK-NEXT:      modelica.while {
// CHECK-NEXT:          %9 = memref.load %3[] : memref<i1>
// CHECK-NEXT:          %10 = scf.if %9 -> (i1) {
// CHECK-NEXT:              %false_2 = constant false
// CHECK-NEXT:              scf.yield %false_2 : i1
// CHECK-NEXT:          } else {
// CHECK-NEXT:              %12 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:              %13 = modelica.constant #modelica.int<10>
// CHECK-NEXT:              %14 = modelica.lt %12, %13 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:              %15 = unrealized_conversion_cast %14 : !modelica.bool to i1
// CHECK-NEXT:              scf.yield %15 : i1
// CHECK-NEXT:          }
// CHECK-NEXT:          %11 = unrealized_conversion_cast %10 : i1 to !modelica.bool
// CHECK-NEXT:          modelica.condition (%11 : !modelica.bool)
// CHECK-NEXT:      } do {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %3[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:      %5 = modelica.constant #modelica.int<5>
// CHECK-NEXT:      %6 = modelica.eq %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:      modelica.if (%6 : !modelica.bool) {
// CHECK-NEXT:          %true = constant true
// CHECK-NEXT:          memref.store %true, %2[] : memref<i1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %7 = memref.load %2[] : memref<i1>
// CHECK-NEXT:      %false_1 = constant false
// CHECK-NEXT:      %8 = cmpi eq, %7, %false_1 : i1
// CHECK-NEXT:      scf.if %8 {
// CHECK-NEXT:          %9 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %10 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          %11 = modelica.add %9, %10 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %0, %11 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          %12 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %13 = modelica.constant #modelica.int<2>
// CHECK-NEXT:          %14 = modelica.add %12, %13 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.member_store %0, %14 : !modelica.member<stack, !modelica.int>
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
            %3 = modelica.constant #modelica.int<10>
            %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
            modelica.condition (%4 : !modelica.bool)
        } do {
            modelica.break
        }
        %2 = modelica.member_load %0 : !modelica.int
        %3 = modelica.constant #modelica.int<5>
        %4 = modelica.eq %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.if (%4 : !modelica.bool) {
            modelica.break
        }
        %5 = modelica.member_load %0 : !modelica.int
        %6 = modelica.constant #modelica.int<1>
        %7 = modelica.add %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %7 : !modelica.member<stack, !modelica.int>
        %8 = modelica.member_load %0 : !modelica.int
        %9 = modelica.constant #modelica.int<2>
        %10 = modelica.add %8, %9 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.member_store %0, %10 : !modelica.member<stack, !modelica.int>
    }
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: %[[X:[a-zA-Z0-9]*]] : !modelica.int
// CHECK-SAME: -> (!modelica.int)
// CHECK-SAME: args_names = ["x"]
// CHECK-SAME: results_names = ["y"]
// CHECK-NEXT: %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT: modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT: modelica.while {
// CHECK-NEXT:      %2 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:      %3 = modelica.constant #modelica.int<10>
// CHECK-NEXT:      %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:      modelica.condition (%4 : !modelica.bool)
// CHECK-NEXT: } do {
// CHECK-NEXT:      %2 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:      %3 = modelica.constant #modelica.int<1>
// CHECK-NEXT:      modelica.store %2[], %3 : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:      %4 = memref.alloca() : memref<i1>
// CHECK-NEXT:      %false = constant false
// CHECK-NEXT:      memref.store %false, %4[] : memref<i1>
// CHECK-NEXT:      modelica.for  {
// CHECK-NEXT:          %5 = memref.load %4[] : memref<i1>
// CHECK-NEXT:          %6 = scf.if %5 -> (i1) {
// CHECK-NEXT:              %false_0 = constant false
// CHECK-NEXT:              scf.yield %false_0 : i1
// CHECK-NEXT:          } else {
// CHECK-NEXT:              %8 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:              %9 = modelica.constant #modelica.int<10>
// CHECK-NEXT:              %10 = modelica.lte %8, %9 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:              %11 = unrealized_conversion_cast %10 : !modelica.bool to i1
// CHECK-NEXT:              scf.yield %11 : i1
// CHECK-NEXT:          }
// CHECK-NEXT:          %7 = unrealized_conversion_cast %6 : i1 to !modelica.bool
// CHECK-NEXT:          modelica.condition (%7 : !modelica.bool)
// CHECK-NEXT:      } body {
// CHECK-NEXT:          %5 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:          %6 = modelica.constant #modelica.int<5>
// CHECK-NEXT:          %7 = modelica.eq %5, %6 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:          modelica.if (%7 : !modelica.bool) {
// CHECK-NEXT:              %true = constant true
// CHECK-NEXT:              memref.store %true, %4[] : memref<i1>
// CHECK-NEXT:          }
// CHECK-NEXT:          %8 = memref.load %4[] : memref<i1>
// CHECK-NEXT:          %false_0 = constant false
// CHECK-NEXT:          %9 = cmpi eq, %8, %false_0 : i1
// CHECK-NEXT:          scf.if %9 {
// CHECK-NEXT:              %10 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:              %11 = modelica.constant #modelica.int<1>
// CHECK-NEXT:              %12 = modelica.add %10, %11 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:              modelica.member_store %0, %11 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:              %13 = modelica.member_load %0 : !modelica.int
// CHECK-NEXT:              %14 = modelica.constant #modelica.int<2>
// CHECK-NEXT:              %15 = modelica.add %13, %14 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:              modelica.member_store %0, %15 : !modelica.member<stack, !modelica.int>
// CHECK-NEXT:          }
// CHECK-NEXT:      } step {
// CHECK-NEXT:          %5 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:          %6 = modelica.constant #modelica.int<1>
// CHECK-NEXT:          %7 = modelica.add %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:          modelica.store %2[], %7 : !modelica.array<stack, !modelica.int>
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
        %2 = modelica.alloca : !modelica.array<stack, !modelica.int>
        %3 = modelica.constant #modelica.int<1>
        modelica.store %2[], %3: !modelica.array<stack, !modelica.int>
        modelica.for {
            %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
            %5 = modelica.constant #modelica.int<10>
            %6 = modelica.lte %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
            modelica.condition (%6 : !modelica.bool)
        } body {
            %4 = modelica.member_load %0 : !modelica.int
            %5 = modelica.constant #modelica.int<5>
            %6 = modelica.eq %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
            modelica.if (%6 : !modelica.bool) {
                modelica.break
            }
            %7 = modelica.member_load %0 : !modelica.int
            %8 = modelica.constant #modelica.int<1>
            %9 = modelica.add %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.member_store %0, %8 : !modelica.member<stack, !modelica.int>
            %10 = modelica.member_load %0 : !modelica.int
            %11 = modelica.constant #modelica.int<2>
            %12 = modelica.add %10, %11 : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.member_store %0, %12 : !modelica.member<stack, !modelica.int>
        } step {
            %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
            %5 = modelica.constant #modelica.int<1>
            %6 = modelica.add %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.store %2[], %6 : !modelica.array<stack, !modelica.int>
        }
    }
}
