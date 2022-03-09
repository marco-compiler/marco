// RUN: modelica-opt %s --split-input-file --convert-modelica-functions --convert-modelica-to-cfg | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%arg0: !modelica.int) -> !modelica.int
// CHECK-NEXT:  %0 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT:  modelica.assignment %1, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %2 = modelica.constant #modelica.int<5>
// CHECK-NEXT:  %3 = modelica.eq %arg0, %2 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %4 = unrealized_conversion_cast %3 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %4, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:  br ^bb3
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:  modelica.assignment %arg0, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %5 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %6 = modelica.constant #modelica.int<2>
// CHECK-NEXT:  %7 = modelica.mul %5, %6 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %7, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb3
// CHECK-NEXT: ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:  %8 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  return %8 : !modelica.int

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
    modelica.function_terminator
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: (%arg0: !modelica.int) -> !modelica.int
// CHECK-NEXT:  %0 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT:  modelica.assignment %1, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:  %2 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %3 = modelica.constant #modelica.int<10>
// CHECK-NEXT:  %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %5 = unrealized_conversion_cast %4 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %5, ^bb2, ^bb5
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:  %6 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %7 = modelica.constant #modelica.int<5>
// CHECK-NEXT:  %8 = modelica.eq %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %9, ^bb3, ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:  br ^bb6
// CHECK-NEXT: ^bb4:  // pred: ^bb2
// CHECK-NEXT:  %10 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %11 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  %12 = modelica.add %10, %11 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %12, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %13 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %14 = modelica.constant #modelica.int<2>
// CHECK-NEXT:  %15 = modelica.mul %13, %14 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %15, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb5:  // pred: ^bb1
// CHECK-NEXT:  br ^bb6
// CHECK-NEXT: ^bb6:  // 2 preds: ^bb3, ^bb5
// CHECK-NEXT:  %16 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  return %16 : !modelica.int

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
        modelica.yield
    }
    modelica.function_terminator
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: (%arg0: !modelica.int) -> !modelica.int
// CHECK-NEXT:  %0 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT:  modelica.assignment %1, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:  %2 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %3 = modelica.constant #modelica.int<10>
// CHECK-NEXT:  %4 = modelica.lt %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %5 = unrealized_conversion_cast %4 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %5, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:  br ^bb3
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:  %6 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %7 = modelica.constant #modelica.int<5>
// CHECK-NEXT:  %8 = modelica.lt %6, %7 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %9 = unrealized_conversion_cast %8 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %9, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT:  br ^bb7
// CHECK-NEXT: ^bb5:  // pred: ^bb3
// CHECK-NEXT:  %10 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %11 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  %12 = modelica.add %10, %11 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %12, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %13 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %14 = modelica.constant #modelica.int<2>
// CHECK-NEXT:  %15 = modelica.mul %13, %14 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %15, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb6:  // pred: ^bb1
// CHECK-NEXT:  br ^bb7
// CHECK-NEXT: ^bb7:  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:  %16 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  return %16 : !modelica.int

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
        modelica.yield
    }
    modelica.function_terminator
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: (%arg0: !modelica.int) -> !modelica.int
// CHECK-NEXT:  %0 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %1 = modelica.constant #modelica.int<0>
// CHECK-NEXT:  modelica.assignment %1, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %2 = modelica.alloca {constant = false} : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %3 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  modelica.store %2[], %3 : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:  %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %5 = modelica.constant #modelica.int<10>
// CHECK-NEXT:  %6 = modelica.lte %4, %5 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %7 = unrealized_conversion_cast %6 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %7, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:  %8 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %9 = modelica.constant #modelica.int<5>
// CHECK-NEXT:  %10 = modelica.lt %8, %9 : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK-NEXT:  %11 = unrealized_conversion_cast %10 : !modelica.bool to i1
// CHECK-NEXT:  cond_br %11, ^bb3, ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:  br ^bb7
// CHECK-NEXT: ^bb4:  // pred: ^bb2
// CHECK-NEXT:  %12 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  modelica.assignment %12, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %13 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %14 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %15 = modelica.add %13, %14 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %15, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb5
// CHECK-NEXT: ^bb5:  // pred: ^bb4
// CHECK-NEXT:  %16 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %17 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  %18 = modelica.add %16, %17 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.store %2[], %18 : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb1
// CHECK-NEXT: ^bb6:  // pred: ^bb1
// CHECK-NEXT:  %19 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %20 = modelica.constant #modelica.int<1>
// CHECK-NEXT:  %21 = modelica.add %19, %20 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %21, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %22 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  %23 = modelica.constant #modelica.int<2>
// CHECK-NEXT:  %24 = modelica.mul %22, %23 : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK-NEXT:  modelica.assignment %24, %0 : !modelica.int, !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  br ^bb7
// CHECK-NEXT: ^bb7:  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:  %25 = modelica.load %0[] : !modelica.array<stack, !modelica.int>
// CHECK-NEXT:  return %25 : !modelica.int

modelica.function @foo(%arg0 : !modelica.int) -> (!modelica.int) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.int>
    %2 = modelica.alloca : !modelica.array<stack, !modelica.int>
    %3 = modelica.constant #modelica.int<1>
    modelica.store %2[], %3 : !modelica.array<stack, !modelica.int>
    modelica.for condition {
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
        modelica.yield
    } step {
        %4 = modelica.load %2[] : !modelica.array<stack, !modelica.int>
        %5 = modelica.constant #modelica.int<1>
        %6 = modelica.add %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %2[], %6 : !modelica.array<stack, !modelica.int>
        modelica.yield
    }
    %4 = modelica.member_load %0 : !modelica.int
    %5 = modelica.constant #modelica.int<1>
    %6 = modelica.add %4, %5 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.member_store %0, %6 : !modelica.member<stack, !modelica.int>
    %7 = modelica.member_load %0 : !modelica.int
    %8 = modelica.constant #modelica.int<2>
    %9 = modelica.mul %7, %8 : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.member_store %0, %9 : !modelica.member<stack, !modelica.int>
    modelica.function_terminator
}
