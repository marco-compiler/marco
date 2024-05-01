// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK:       bmodelica.raw_function @foo
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.algorithm {
        bmodelica.return
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool) {
// CHECK:           cf.cond_br %{{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:       bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.if (%0 : !bmodelica.bool) {
            bmodelica.return
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool) {
// CHECK:           cf.cond_br %{{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out:.*]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.while {
            bmodelica.condition (%0 : !bmodelica.bool)
        } do {
            bmodelica.return
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.bool) {
// CHECK:           cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.bool, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.while {
            bmodelica.condition (%0 : !bmodelica.bool)
        } do {
            %1 = bmodelica.variable_get @y : !bmodelica.bool

            bmodelica.if (%1 : !bmodelica.bool) {
                bmodelica.return
            }

            bmodelica.print %1 : !bmodelica.bool
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.bool, %[[z:.*]]: !bmodelica.bool) {
// CHECK:           cf.br ^[[while_1_condition:.*]]
// CHECK-NEXT:  ^[[while_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[while_2_condition:.*]], ^[[while_1_out:.*]]
// CHECK-NEXT:  ^[[while_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[while_2_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br {{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[while_2_condition]]
// CHECK-NEXT:  ^[[while_2_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_1_condition]]
// CHECK-NEXT:  ^[[while_1_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.bool, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.while {
            bmodelica.condition (%0 : !bmodelica.bool)
        } do {
            %1 = bmodelica.variable_get @y : !bmodelica.bool

            bmodelica.while {
                bmodelica.condition (%1 : !bmodelica.bool)
            } do {
                %2 = bmodelica.variable_get @z : !bmodelica.bool

                bmodelica.if (%2 : !bmodelica.bool) {
                    bmodelica.return
                }

                bmodelica.print %2 : !bmodelica.bool
            }

            bmodelica.print %1 : !bmodelica.bool
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.int, %[[z:.*]]: !bmodelica.int) {
// CHECK:           cf.cond_br %{{.*}}, ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.for condition {
            bmodelica.condition (%0 : !bmodelica.bool)
        } body {
            %1 = bmodelica.variable_get @y : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.return
        } step {
            %1 = bmodelica.variable_get @z : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.yield
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.bool, %[[z:.*]]: !bmodelica.int) {
// CHECK:           cf.br ^[[for_condition:.*]]
// CHECK:       ^[[for_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.for condition {
            bmodelica.condition (%0 : !bmodelica.bool)
        } body {
            %1 = bmodelica.variable_get @y : !bmodelica.bool

            bmodelica.if (%1 : !bmodelica.bool) {
                bmodelica.return
            }

            bmodelica.print %1 : !bmodelica.bool
            bmodelica.yield
        } step {
            %1 = bmodelica.variable_get @z : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.yield
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: !bmodelica.bool, %[[y:.*]]: !bmodelica.bool, %[[z:.*]]: !bmodelica.bool, %[[t:.*]]: !bmodelica.int, %[[k:.*]]: !bmodelica.int) {
// CHECK:           cf.br ^[[for_1_condition:.*]]
// CHECK:       ^[[for_1_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[for_2_condition:.*]], ^[[for_1_out:.*]]
// CHECK-NEXT:  ^[[for_2_condition]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[if:.*]], ^[[for_2_out:.*]]
// CHECK-NEXT:  ^[[if]]:
// CHECK:           cf.cond_br %{{.*}}, ^[[out:.*]], ^[[if_out:.*]]
// CHECK:       ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      bmodelica.print %[[t]]
// CHECK-NEXT:      cf.br ^[[for_2_condition]]
// CHECK-NEXT:  ^[[for_2_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      bmodelica.print %[[k]]
// CHECK-NEXT:      cf.br ^[[for_1_condition]]
// CHECK-NEXT:  ^[[for_1_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.bool, input>
    bmodelica.variable @t : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @k : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.bool

        bmodelica.for condition {
            bmodelica.condition (%0 : !bmodelica.bool)
        } body {
            %1 = bmodelica.variable_get @y : !bmodelica.bool

            bmodelica.for condition {
                bmodelica.condition (%1 : !bmodelica.bool)
            } body {
                %2 = bmodelica.variable_get @z : !bmodelica.bool

                bmodelica.if (%2 : !bmodelica.bool) {
                    bmodelica.return
                }

                bmodelica.print %2 : !bmodelica.bool
                bmodelica.yield
            } step {
                %2 = bmodelica.variable_get @t : !bmodelica.int
                bmodelica.print %2 : !bmodelica.int
                bmodelica.yield
            }

            bmodelica.print %1 : !bmodelica.bool
            bmodelica.yield
        } step {
            %1 = bmodelica.variable_get @k : !bmodelica.int
            bmodelica.print %1 : !bmodelica.int
            bmodelica.yield
        }

        bmodelica.print %0 : !bmodelica.bool
    }
}
