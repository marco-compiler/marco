// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cfg | FileCheck %s

// CHECK:      func @foo() {
// CHECK:           br ^[[while_condition:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[while_condition]]:
// CHECK:           cond_br %{{[a-zA-Z0-9]*}}, ^[[while_body:[a-zA-Z0-9]*]], ^[[while_out:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[while_body]]:
// CHECK:           br ^[[while_condition]]
// CHECK-NEXT: ^[[while_out]]:
// CHECK:           br ^[[out:[a-zA-Z0-9]*]]
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
