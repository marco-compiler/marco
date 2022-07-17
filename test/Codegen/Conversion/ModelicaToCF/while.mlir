// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:      func @foo() {
// CHECK:           br ^[[while_condition:.*]]
// CHECK-NEXT: ^[[while_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT: ^[[while_body]]:
// CHECK:           br ^[[while_condition]]
// CHECK-NEXT: ^[[while_out]]:
// CHECK:           br ^[[out:.*]]
// CHECK-NEXT: ^[[out]]:
// CHECK:           return
// CHECK-NEXT: }

modelica.function @foo : () -> () {
    modelica.while {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } do {

    }
}
