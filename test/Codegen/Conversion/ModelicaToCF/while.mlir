// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:      modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.int) {
// CHECK:           br ^[[while_condition:.*]]
// CHECK-NEXT: ^[[while_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT: ^[[while_body]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      br ^[[while_condition]]
// CHECK-NEXT: ^[[while_out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT: }

modelica.function @foo {
    modelica.variable @x : !modelica.member<!modelica.bool, input>
    modelica.variable @y : !modelica.member<!modelica.int, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.while {
            modelica.condition (%0 : !modelica.bool)
        } do {
            %1 = modelica.variable_get @y : !modelica.int
            modelica.print %1 : !modelica.int
        }

        modelica.print %0 : !modelica.bool
    }
}
