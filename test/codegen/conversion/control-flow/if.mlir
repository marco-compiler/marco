// RUN: modelica-opt %s --convert-to-cfg | FileCheck %s

// CHECK-LABEL: @if
// CHECK: cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:   // pred: ^bb0
// CHECK-NEXT:   %{{.*}} = modelica.constant #modelica.int<0>
// CHECK-NEXT:   br ^bb2
// CHECK-NEXT: ^bb2:   // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   modelica.return

modelica.function @if(%arg0 : !modelica.bool) -> () attributes {args_names = ["condition"], results_names = []} {
    modelica.if (%arg0 : !modelica.bool) {
        %c0 = modelica.constant #modelica.int<0>
        modelica.yield
    }

    modelica.return
}

// CHECK-LABEL: @ifElse
// CHECK: cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:   // pred: ^bb0
// CHECK-NEXT:   %{{.*}} = modelica.constant #modelica.int<0>
// CHECK-NEXT:   br ^bb3
// CHECK-NEXT: ^bb2:   // pred: ^bb0
// CHECK-NEXT:   %{{.*}} = modelica.constant #modelica.int<1>
// CHECK-NEXT:   br ^bb3
// CHECK-NEXT: ^bb3:   // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:   modelica.return

modelica.function @ifElse(%arg0 : !modelica.bool) -> () attributes {args_names = ["condition"], results_names = []} {
    modelica.if (%arg0 : !modelica.bool) {
        %c0 = modelica.constant #modelica.int<0>
        modelica.yield
    } else {
        %c1 = modelica.constant #modelica.int<1>
        modelica.yield
    }

    modelica.return
}
