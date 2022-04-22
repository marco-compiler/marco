// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cfg | FileCheck %s

// CHECK:       func @foo(%arg0: !modelica.int) -> !modelica.int {
// CHECK:           cond_br %{{[a-zA-Z0-9]*}}, ^[[if_then:[a-zA-Z0-9]*]], ^[[if_else:[a-zA-Z0-9]*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           modelica.constant #modelica.int<1>
// CHECK:           modelica.store
// CHECK:           br ^[[if_out:[a-zA-Z0-9]*]]
// CHECK-NEXT:  ^[[if_else]]:
// CHECK:           modelica.constant #modelica.int<2>
// CHECK:           modelica.store
// CHECK:           br ^[[if_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      br ^[[out:[a-zA-Z0-9]*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           return
// CHECK-NEXT:  }

modelica.function @foo : (!modelica.int) -> (!modelica.int) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int, output>

    %2 = modelica.member_load %0 : !modelica.member<!modelica.int, input>
    %3 = modelica.constant #modelica.int<0>

    %4 = modelica.eq %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool

    modelica.if (%4 : !modelica.bool) {
        %5 = modelica.constant #modelica.int<1>
        modelica.member_store %1, %5 : !modelica.member<!modelica.int, output>
    } else {
        %5 = modelica.constant #modelica.int<2>
        modelica.member_store %1, %5 : !modelica.member<!modelica.int, output>
    }
}
