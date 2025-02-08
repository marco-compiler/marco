// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf | FileCheck %s

// CHECK-LABEL: @while
// CHECK-SAME:  (%[[x:.*]]: i1, %[[y:.*]]: i64)
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      br ^[[while_condition:.*]]
// CHECK-NEXT: ^[[while_condition]]:
// CHECK-NEXT:      cond_br %[[x]], ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT: ^[[while_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_condition]]
// CHECK-NEXT: ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT: }

bmodelica.function @while {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i64, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.while {
            bmodelica.condition (%0 : i1)
        } do {
            %1 = bmodelica.variable_get @y : i64
            bmodelica.print %1 : i64
        }

        bmodelica.print %0 : i1
    }
}
