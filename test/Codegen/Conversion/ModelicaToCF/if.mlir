// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.int) -> !modelica.int {
// CHECK:           %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_else:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           %[[then_value:.*]] = modelica.constant #modelica.int<1>
// CHECK:           modelica.store %[[y]][], %[[then_value]]
// CHECK:           br ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_else]]:
// CHECK:           %[[else_value:.*]] = modelica.constant #modelica.int<2>
// CHECK:           modelica.store %[[y]][], %[[else_value]]
// CHECK:           br ^[[if_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
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
