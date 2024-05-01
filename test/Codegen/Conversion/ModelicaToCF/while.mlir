// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:      bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.int) {
// CHECK:           br ^[[while_condition:.*]]
// CHECK-NEXT: ^[[while_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT: ^[[while_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      br ^[[while_condition]]
// CHECK-NEXT: ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT: }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.while {
            bmodelica.condition (%0 : !bmodelica.bool)
        } do {
            %1 = bmodelica.variable_get @y : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}
