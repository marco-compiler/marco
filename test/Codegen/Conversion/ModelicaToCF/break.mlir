// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:       modelica.raw_function @foo() {
// CHECK:           br ^[[while_condition:.*]]
// CHECK:       ^[[while_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body:.*]]:
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           br ^[[while_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK:           br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.while {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } do {
        %0 = modelica.constant #modelica.bool<true>

        modelica.if (%0 : !modelica.bool) {
            modelica.break
        }
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           br ^[[while_1_condition:.*]]
// CHECK:       ^[[while_1_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_1_body:.*]], ^[[while_1_out:.*]]
// CHECK-NEXT:  ^[[while_1_body]]:
// CHECK:           br ^[[while_2_condition:.*]]
// CHECK:       ^[[while_2_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_2_body:.*]], ^[[while_2_out:.*]]
// CHECK-NEXT:  ^[[while_2_body]]:
// CHECK:           br ^[[while_2_out]]
// CHECK-NEXT:  ^[[while_2_out]]:
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           br ^[[while_1_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           br ^[[while_1_condition]]
// CHECK-NEXT:  ^[[while_1_out]]:
// CHECK:           br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.while {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } do {
        modelica.while {
            %0 = modelica.constant #modelica.bool<true>
            modelica.condition (%0 : !modelica.bool)
        } do {
            modelica.break
        }

        %0 = modelica.constant #modelica.bool<true>

        modelica.if (%0 : !modelica.bool) {
            modelica.break
        }
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           br ^[[while_condition:.*]]
// CHECK:       ^[[while_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK:           br ^[[for_condition:.*]]
// CHECK-NEXT:  ^[[for_condition]]:
// CHECK:           cond_br %{{.*}}, ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK:           cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           br ^[[for_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           br ^[[for_step:.*]]
// CHECK-NEXT:  ^[[for_step]]:
// CHECK:           br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK:           br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK-NEXT:      br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.while {
        %3 = modelica.constant #modelica.bool<true>
        modelica.condition (%3 : !modelica.bool)
    } do {
        modelica.for condition {
            %3 = modelica.constant #modelica.bool<true>
            modelica.condition (%3 : !modelica.bool)
        } body {
            %3 = modelica.constant #modelica.bool<true>

            modelica.if (%3 : !modelica.bool) {
                modelica.break
            }

            modelica.yield
        } step {
            modelica.yield
        }
    }
}
