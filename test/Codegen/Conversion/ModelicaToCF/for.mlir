// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:      modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.int, %[[z:.*]]: !modelica.int) {
// CHECK:           br ^[[for_condition:.*]]
// CHECK-NEXT: ^[[for_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[for_body:.*]], ^[[out:.*]]
// CHECK-NEXT: ^[[for_body]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      modelica.print %[[z]]
// CHECK-NEXT:      br ^[[for_condition]]
// CHECK-NEXT: ^[[out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT: }

modelica.function @foo {
    modelica.variable @x : !modelica.member<!modelica.bool, input>
    modelica.variable @y : !modelica.member<!modelica.int, input>
    modelica.variable @z : !modelica.member<!modelica.int, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.for condition {
            modelica.condition (%0 : !modelica.bool)
        } body {
            %1 = modelica.variable_get @y : !modelica.int
            modelica.print %1 : !modelica.int
            modelica.yield
        } step {
            %1 = modelica.variable_get @z : !modelica.int
            modelica.print %1 : !modelica.int
            modelica.yield
        }

        modelica.print %0 : !modelica.bool
    }
}
