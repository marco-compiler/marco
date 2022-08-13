// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.return
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_out:.*]]:
// CHECK:           cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    %0 = modelica.constant #modelica.bool<true>

    modelica.if (%0 : !modelica.bool) {
        modelica.return
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[while_condition:.*]]
// CHECK:       ^[[while_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK:           cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.while {
        %0 = modelica.constant #modelica.bool<true> : !modelica.bool
        modelica.condition (%0 : !modelica.bool)
    } do {
        modelica.return
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[while_condition:.*]]
// CHECK:       ^[[while_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK:           cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.while {
        %0 = modelica.constant #modelica.bool<true> : !modelica.bool
        modelica.condition (%0 : !modelica.bool)
    } do {
        %0 = modelica.constant #modelica.bool<true>

        modelica.if (%0 : !modelica.bool) {
            modelica.return
        }
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[while_1_condition:.*]]
// CHECK:       ^[[while_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_1_body:.*]], ^[[while_1_out:.*]]
// CHECK-NEXT:  ^[[while_1_body]]:
// CHECK:           cf.br ^[[while_2_condition:.*]]
// CHECK-NEXT:  ^[[while_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_2_body:.*]], ^[[while_2_out:.*]]
// CHECK-NEXT:  ^[[while_2_body]]:
// CHECK:           cf.cond_br {{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           cf.br ^[[while_2_condition:.*]]
// CHECK-NEXT:  ^[[while_2_out]]:
// CHECK:           cf.br ^[[while_1_condition]]
// CHECK-NEXT:  ^[[while_1_out]]:
// CHECK:           cf.br ^[[out:.*]]
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
            %0 = modelica.constant #modelica.bool<true>

            modelica.if (%0 : !modelica.bool) {
                modelica.return
            }
        }
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[for_condition:.*]]
// CHECK:       ^[[for_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.for condition {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } body {
        modelica.return
    } step {
        modelica.yield
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[for_condition:.*]]
// CHECK:       ^[[for_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           cf.br ^[[for_step:.*]]
// CHECK-NEXT:  ^[[for_step]]:
// CHECK-NEXT:      cf.br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.for condition {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } body {
        %0 = modelica.constant #modelica.bool<true>

        modelica.if (%0 : !modelica.bool) {
            modelica.return
        }

        modelica.yield
    } step {
        modelica.yield
    }
}

// -----

// CHECK:       modelica.raw_function @foo() {
// CHECK:           cf.br ^[[for_1_condition:.*]]
// CHECK:       ^[[for_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_1_body:.*]], ^[[for_1_out:.*]]
// CHECK-NEXT:  ^[[for_1_body]]:
// CHECK:           cf.br ^[[for_2_condition:.*]]
// CHECK-NEXT:  ^[[for_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_2_body:.*]], ^[[for_2_out:.*]]
// CHECK-NEXT:  ^[[for_2_body]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if_then:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_then]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK:       ^[[if_out]]:
// CHECK:           cf.br ^[[for_2_step:.*]]
// CHECK-NEXT:  ^[[for_2_step]]:
// CHECK:           cf.br ^[[for_2_condition]]
// CHECK-NEXT:  ^[[for_2_out]]:
// CHECK:           cf.br ^[[for_1_step:.*]]
// CHECK-NEXT:  ^[[for_1_step]]:
// CHECK-NEXT:      cf.br ^[[for_1_condition]]
// CHECK-NEXT:  ^[[for_1_out]]:
// CHECK:           cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : () -> () {
    modelica.for condition {
        %0 = modelica.constant #modelica.bool<true>
        modelica.condition (%0 : !modelica.bool)
    } body {
        modelica.for condition {
            %0 = modelica.constant #modelica.bool<true>
            modelica.condition (%0 : !modelica.bool)
        } body {
            %0 = modelica.constant #modelica.bool<true>

            modelica.if (%0 : !modelica.bool) {
                modelica.return
            }

            modelica.yield
        } step {
            modelica.yield
        }

        modelica.yield
    } step {
        modelica.yield
    }
}
