// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:      bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.int, %[[z:.*]]: !bmodelica.int) {
// CHECK:           br ^[[for_condition:.*]]
// CHECK-NEXT: ^[[for_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[for_body:.*]], ^[[out:.*]]
// CHECK-NEXT: ^[[for_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      br ^[[for_condition]]
// CHECK-NEXT: ^[[out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT: }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.for condition {
            bmodelica.condition (%0 : !bmodelica.bool)
        } body {
            %1 = bmodelica.variable_get @y : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.yield
        } step {
            %1 = bmodelica.variable_get @z : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.yield
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}
