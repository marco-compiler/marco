// RUN: modelica-opt %s --convert-to-cfg | FileCheck %s

// CHECK-LABEL: @breakable_while
// CHECK-NEXT: %[[BREAK:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.ptr<{{.*}}, !modelica.bool>
// CHECK-NEXT: %[[RETURN:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.ptr<{{.*}}, !modelica.bool>
// CHECK: br ^bb1
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb6
// CHECK: %[[BREAK_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[BREAK]][] : !modelica.ptr<{{.*}}, !modelica.bool>
// CHECK: %[[BREAK_COND:[a-zA-Z0-9]*]] = unrealized_conversion_cast %[[BREAK_LOAD]] : !modelica.bool to i1
// CHECK: %[[RETURN_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[RETURN]][] : !modelica.ptr<{{.*}}, !modelica.bool>
// CHECK: %[[RETURN_COND:[a-zA-Z0-9]*]] = unrealized_conversion_cast %[[RETURN_LOAD]] : !modelica.bool to i1
// CHECK: %[[OR_OP:[a-zA-Z0-9]*]] = or %[[BREAK_COND]], %[[RETURN_COND]] : i1
// CHECK: %[[TRUE:[a-zA-Z0-9]*]] = constant true
// CHECK: %[[STOP_CONDITION:[a-zA-Z0-9]*]] = cmpi eq, %[[OR_OP]], %[[TRUE]] : i1
// CHECK: cond_br %[[STOP_CONDITION]], ^bb2, ^bb3
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %[[FALSE:[a-zA-Z0-9]*]] = constant false
// CHECK-NEXT: br ^bb4(%[[FALSE]] : i1)
// CHECK-NEXT: ^bb3:  // pred: ^bb1
// CHECK: br ^bb4(%{{.*}} : i1)
// CHECK-NEXT: ^bb4(%[[OVERALL_CONDITION:[a-zA-Z0-9]*]]: i1):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT: br ^bb5
// CHECK-NEXT: ^bb5:  // pred: ^bb4
// CHECK-NEXT: cond_br %[[OVERALL_CONDITION]], ^bb6, ^bb7
// CHECK-NEXT: ^bb6:  // pred: ^bb5
// CHECK: %{{.*}} = modelica.constant #modelica.int<0>
// CHECK: br ^bb1
// CHECK-NEXT: ^bb7:  // pred: ^bb5
// CHECK: modelica.return

modelica.function @breakable_while(%arg0 : !modelica.bool) -> () attributes {args_names = ["condition"], results_names = []} {
    %break = modelica.alloca : !modelica.ptr<stack, !modelica.bool>
    %return = modelica.alloca : !modelica.ptr<stack, !modelica.bool>

    modelica.breakable_while (break = %break : !modelica.ptr<stack, !modelica.bool>, return = %return : !modelica.ptr<stack, !modelica.bool>) {
        modelica.condition (%arg0 : !modelica.bool)
    } do {
        %c0 = modelica.constant #modelica.int<0>
        modelica.yield
    }

    modelica.return
}
