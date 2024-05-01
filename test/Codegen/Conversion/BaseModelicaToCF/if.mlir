// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:       bmodelica.raw_function @foo(%{{.*}}: !bmodelica.int) -> !bmodelica.int {
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_else:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK-NEXT:      bmodelica.raw_variable_set
// CHECK-NEXT:      bmodelica.print
// CHECK-NEXT:      br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_else]]:
// CHECK-NEXT:      bmodelica.raw_variable_set
// CHECK-NEXT:      br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, output>

    bmodelica.algorithm {
        %2 = bmodelica.variable_get @x : !bmodelica.int
        %3 = bmodelica.constant #bmodelica.int<0>

        %4 = bmodelica.eq %2, %3 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool

        bmodelica.if (%4 : !bmodelica.bool) {
            %5 = bmodelica.constant #bmodelica.int<1>
            bmodelica.variable_set @y, %5 : !bmodelica.int
            bmodelica.print %5 : !bmodelica.int
        } else {
            %5 = bmodelica.constant #bmodelica.int<2>
            bmodelica.variable_set @y, %5 : !bmodelica.int
        }
    }
}
