// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:      func @foo() {
// CHECK:           br ^[[for_condition:.*]]
// CHECK-NEXT: ^[[for_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT: ^[[for_body]]:
// CHECK:           br ^[[for_step:.*]]
// CHECK-NEXT: ^[[for_step]]:
// CHECK:           br ^[[for_condition]]
// CHECK-NEXT: ^[[for_out]]:
// CHECK:           br ^[[out:.*]]
// CHECK-NEXT: ^[[out]]:
// CHECK:           return
// CHECK-NEXT: }

modelica.function @foo : () -> () {
    modelica.for condition {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } body {
        modelica.yield
    } step {
        modelica.yield
    }
}
