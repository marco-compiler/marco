// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool) {
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.while {
            modelica.condition (%0 : !modelica.bool)
        } do {
            modelica.break
        }

        modelica.print %0 : !modelica.bool
    }
}

// -----

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.bool) {
// CHECK:           cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[out]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK:           modelica.print %[[y]]
// CHECK:           cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      modelica.print
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>
    modelica.variable @y : !modelica.variable<!modelica.bool, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.while {
            modelica.condition (%0 : !modelica.bool)
        } do {
            %1 = modelica.variable_get @y : !modelica.bool

            modelica.if (%1 : !modelica.bool) {
                modelica.break
            }

            modelica.print %1 : !modelica.bool
        }

        modelica.print %0 : !modelica.bool
    }
}

// -----

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.bool, %[[z:.*]]: !modelica.bool) {
// CHECK:           cf.br ^[[while_1_condition:.*]]
// CHECK-NEXT:  ^[[while_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_2_condition:.*]], ^[[out:.*]]
// CHECK-NEXT:  ^[[while_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[while_2_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_2_out]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      modelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[while_2_condition]]
// CHECK-NEXT:  ^[[while_2_out]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_1_condition]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>
    modelica.variable @y : !modelica.variable<!modelica.bool, input>
    modelica.variable @z : !modelica.variable<!modelica.bool, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.while {
            modelica.condition (%0 : !modelica.bool)
        } do {
            %1 = modelica.variable_get @y : !modelica.bool

            modelica.while {
                modelica.condition (%1 : !modelica.bool)
            } do {
                %2 = modelica.variable_get @z : !modelica.bool

                modelica.if (%2 : !modelica.bool) {
                    modelica.break
                }

                modelica.print %2 : !modelica.bool
            }

            modelica.print %1 : !modelica.bool
        }

        modelica.print %0 : !modelica.bool
    }
}

// -----

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.int, %[[z:.*]]: !modelica.int) {
// CHECK:           cf.cond_br %{{.*}}, ^[[for_body:.*]], ^[[out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>
    modelica.variable @y : !modelica.variable<!modelica.int, input>
    modelica.variable @z : !modelica.variable<!modelica.int, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.for condition {
            modelica.condition (%0 : !modelica.bool)
        } body {
            %1 = modelica.variable_get @y : !modelica.int
            modelica.print %1 : !modelica.int
            modelica.break
        } step {
            %1 = modelica.variable_get @z : !modelica.int
            modelica.print %1 : !modelica.int
            modelica.yield
        }

        modelica.print %0 : !modelica.bool
    }
}

// -----

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.bool, %[[z:.*]]: !modelica.bool, %[[t:.*]]: !modelica.int) {
// CHECK:           cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_condition:.*]], ^[[out:.*]]
// CHECK-NEXT:  ^[[for_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      modelica.print %[[z]]
// CHECK-NEXT:      modelica.print %[[t]]
// CHECK-NEXT:      cf.br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>
    modelica.variable @y : !modelica.variable<!modelica.bool, input>
    modelica.variable @z : !modelica.variable<!modelica.bool, input>
    modelica.variable @t : !modelica.variable<!modelica.int, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.while {
            modelica.condition (%0 : !modelica.bool)
        } do {
            %1 = modelica.variable_get @y : !modelica.bool

            modelica.for condition {
                modelica.condition (%1 : !modelica.bool)
            } body {
                %2 = modelica.variable_get @z : !modelica.bool

                modelica.if (%2 : !modelica.bool) {
                    modelica.break
                }

                modelica.print %2 : !modelica.bool
                modelica.yield
            } step {
                %2 = modelica.variable_get @t : !modelica.int
                modelica.print %2 : !modelica.int
                modelica.yield
            }

            modelica.print %1 : !modelica.bool
        }

        modelica.print %0 : !modelica.bool
    }
}

// -----

// CHECK:       modelica.raw_function @foo(%[[x:.*]]: !modelica.bool, %[[y:.*]]: !modelica.bool, %[[z:.*]]: !modelica.bool, %[[t:.*]]: !modelica.int, %[[k:.*]]: !modelica.int, %[[l:.*]]: !modelica.int) {
// CHECK:           cf.br ^[[for_1_condition:.*]]
// CHECK-NEXT:  ^[[for_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_2_condition:.*]], ^[[out:.*]]
// CHECK-NEXT:  ^[[for_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[for_2_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_2_out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      modelica.print %[[z]]
// CHECK-NEXT:      modelica.print %[[t]]
// CHECK-NEXT:      cf.br ^[[for_2_condition]]
// CHECK-NEXT:  ^[[for_2_out]]:
// CHECK-NEXT:      modelica.print %[[y]]
// CHECK-NEXT:      modelica.print %[[k]]
// CHECK-NEXT:      cf.br ^[[for_1_condition]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.bool, input>
    modelica.variable @y : !modelica.variable<!modelica.bool, input>
    modelica.variable @z : !modelica.variable<!modelica.bool, input>
    modelica.variable @t : !modelica.variable<!modelica.int, input>
    modelica.variable @k : !modelica.variable<!modelica.int, input>
    modelica.variable @l : !modelica.variable<!modelica.int, input>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.bool

        modelica.for condition {
            modelica.condition (%0 : !modelica.bool)
        } body {
            %1 = modelica.variable_get @y : !modelica.bool

            modelica.for condition {
                modelica.condition (%1 : !modelica.bool)
            } body {
                %2 = modelica.variable_get @z : !modelica.bool

                modelica.if (%2 : !modelica.bool) {
                    modelica.break
                }

                modelica.print %2 : !modelica.bool
                modelica.yield
            } step {
                %2 = modelica.variable_get @t : !modelica.int
                modelica.print %2 : !modelica.int
                modelica.yield
            }

            modelica.print %1 : !modelica.bool
            modelica.yield
        } step {
            %1 = modelica.variable_get @k : !modelica.int
            modelica.print %1 : !modelica.int
            modelica.yield
        }

        modelica.print %0 : !modelica.bool
    }
}
