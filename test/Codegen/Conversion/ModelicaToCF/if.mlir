// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.int) -> !modelica.int {
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_else:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK-NEXT:      modelica.raw_variable_set
// CHECK-NEXT:      modelica.print
// CHECK-NEXT:      br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_else]]:
// CHECK-NEXT:      modelica.raw_variable_set
// CHECK-NEXT:      br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
    modelica.variable @y : !modelica.variable<!modelica.int, output>

    modelica.algorithm {
        %2 = modelica.variable_get @x : !modelica.int
        %3 = modelica.constant #modelica.int<0>

        %4 = modelica.eq %2, %3 : (!modelica.int, !modelica.int) -> !modelica.bool

        modelica.if (%4 : !modelica.bool) {
            %5 = modelica.constant #modelica.int<1>
            modelica.variable_set @y, %5 : !modelica.int
            modelica.print %5 : !modelica.int
        } else {
            %5 = modelica.constant #modelica.int<2>
            modelica.variable_set @y, %5 : !modelica.int
        }
    }
}
