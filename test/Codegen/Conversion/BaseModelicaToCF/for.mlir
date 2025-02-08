// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf | FileCheck %s

// CHECK-LABEL: @for
// CHECK-SAME: (%[[x:.*]]: i1, %[[y:.*]]: i64, %[[z:.*]]: i64)
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[for_condition:.*]]
// CHECK-NEXT:  ^[[for_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[for_step:.*]]
// CHECK-NEXT:  ^[[for_step]]:
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT: }

bmodelica.function @for {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i64, input>
    bmodelica.variable @z : !bmodelica.variable<i64, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.for condition {
            bmodelica.condition (%0 : i1)
        } body {
            %1 = bmodelica.variable_get @y : i64
            bmodelica.print %1 : i64
            bmodelica.yield
        } step {
            %1 = bmodelica.variable_get @z : i64
            bmodelica.print %1 : i64
            bmodelica.yield
        }

        bmodelica.print %0 : i1
    }
}
