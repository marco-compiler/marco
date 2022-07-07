// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:      func @foo() {
// CHECK:           br ^[[for_condition:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[for_condition]]:
// CHECK:           cond_br %{{[a-zA-Z0-9]*}}, ^[[for_body:[a-zA-Z0-9]*]], ^[[for_out:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[for_body]]:
// CHECK:           br ^[[for_step:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[for_step]]:
// CHECK:           br ^[[for_condition]]
// CHECK-NEXT: ^[[for_out]]:
// CHECK:           br ^[[out:[a-zA-Z0-9]*]]
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
