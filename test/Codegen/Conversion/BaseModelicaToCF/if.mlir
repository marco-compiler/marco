// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:       bmodelica.raw_function @foo(%{{.*}}: i64) -> i64 {
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
    bmodelica.variable @x : !bmodelica.variable<i64, input>
    bmodelica.variable @y : !bmodelica.variable<i64, output>

    bmodelica.algorithm {
        %2 = bmodelica.variable_get @x : i64
        %3 = arith.constant 0 : i64

        %4 = bmodelica.eq %2, %3 : (i64, i64) -> i1

        bmodelica.if (%4 : i1) {
            %5 = arith.constant 1 : i64
            bmodelica.variable_set @y, %5 : i64
            bmodelica.print %5 : i64
        } else {
            %5 = arith.constant 2 : i64
            bmodelica.variable_set @y, %5 : i64
        }
    }
}
