// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf | FileCheck %s

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK-NEXT:      cf.br ^[[while_out]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<i1, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.while {
            bmodelica.condition (%0 : i1)
        } do {
            bmodelica.break
        }

        bmodelica.print %0 : i1
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1, %[[y:.*]]: i1) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK-NEXT:      cf.cond_br %[[y]], ^[[if_true:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_true]]:
// CHECK-NEXT:      cf.br ^[[while_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i1, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.while {
            bmodelica.condition (%0 : i1)
        } do {
            %1 = bmodelica.variable_get @y : i1

            bmodelica.if (%1 : i1) {
                bmodelica.break
            }

            bmodelica.print %1 : i1
        }

        bmodelica.print %0 : i1
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1, %[[y:.*]]: i1, %[[z:.*]]: i1) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[while_1_condition:.*]]
// CHECK-NEXT:  ^[[while_1_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[while_1_body:.*]], ^[[while_1_out:.*]]
// CHECK-NEXT:  ^[[while_1_body]]:
// CHECK-NEXT:      cf.br ^[[while_2_condition:.*]]
// CHECK-NEXT:  ^[[while_2_condition]]:
// CHECK-NEXT:      cf.cond_br %[[y]], ^[[while_2_body:.*]], ^[[while_2_out:.*]]
// CHECK-NEXT:  ^[[while_2_body]]:
// CHECK-NEXT:      cf.cond_br %[[z]], ^[[if_true:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_true]]:
// CHECK-NEXT:      cf.br ^[[while_2_out]]
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
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i1, input>
    bmodelica.variable @z : !bmodelica.variable<i1, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.while {
            bmodelica.condition (%0 : i1)
        } do {
            %1 = bmodelica.variable_get @y : i1

            bmodelica.while {
                bmodelica.condition (%1 : i1)
            } do {
                %2 = bmodelica.variable_get @z : i1

                bmodelica.if (%2 : i1) {
                    bmodelica.break
                }

                bmodelica.print %2 : i1
            }

            bmodelica.print %1 : i1
        }

        bmodelica.print %0 : i1
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1, %[[y:.*]]: i64, %[[z:.*]]: i64) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[for_condition:.*]]
// CHECK-NEXT:  ^[[for_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i64, input>
    bmodelica.variable @z : !bmodelica.variable<i64, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.for condition {
            bmodelica.condition (%0 : i1)
        } body {
            %1 = bmodelica.variable_get @y : i64
            bmodelica.print %1 : i64
            bmodelica.break
        } step {
            %1 = bmodelica.variable_get @z : i64
            bmodelica.print %1 : i64
            bmodelica.yield
        }

        bmodelica.print %0 : i1
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1, %[[y:.*]]: i1, %[[z:.*]]: i1, %[[t:.*]]: i64) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[while_condition:.*]]
// CHECK-NEXT:  ^[[while_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[while_body:.*]], ^[[while_out:.*]]
// CHECK-NEXT:  ^[[while_body]]:
// CHECK-NEXT:      cf.br ^[[for_condition:.*]]
// CHECK-NEXT:  ^[[for_condition]]:
// CHECK-NEXT:      cf.cond_br %[[y]], ^[[for_body:.*]], ^[[for_out:.*]]
// CHECK-NEXT:  ^[[for_body]]:
// CHECK-NEXT:      cf.cond_br %[[z]], ^[[if_true:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_true]]:
// CHECK-NEXT:      cf.br ^[[for_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[for_step:.*]]
// CHECK-NEXT:  ^[[for_step]]:
// CHECK-NEXT:      bmodelica.print %[[t]]
// CHECK-NEXT:      cf.br ^[[for_condition]]
// CHECK-NEXT:  ^[[for_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[while_condition]]
// CHECK-NEXT:  ^[[while_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i1, input>
    bmodelica.variable @z : !bmodelica.variable<i1, input>
    bmodelica.variable @t : !bmodelica.variable<i64, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.while {
            bmodelica.condition (%0 : i1)
        } do {
            %1 = bmodelica.variable_get @y : i1

            bmodelica.for condition {
                bmodelica.condition (%1 : i1)
            } body {
                %2 = bmodelica.variable_get @z : i1

                bmodelica.if (%2 : i1) {
                    bmodelica.break
                }

                bmodelica.print %2 : i1
                bmodelica.yield
            } step {
                %2 = bmodelica.variable_get @t : i64
                bmodelica.print %2 : i64
                bmodelica.yield
            }

            bmodelica.print %1 : i1
        }

        bmodelica.print %0 : i1
    }
}

// -----

// CHECK:       bmodelica.raw_function @foo(%[[x:.*]]: i1, %[[y:.*]]: i1, %[[z:.*]]: i1, %[[t:.*]]: i64, %[[k:.*]]: i64, %[[l:.*]]: i64) {
// CHECK-NEXT:      cf.br ^[[body:.*]]
// CHECK-NEXT:  ^[[body]]:
// CHECK-NEXT:      cf.br ^[[for_1_condition:.*]]
// CHECK-NEXT:  ^[[for_1_condition]]:
// CHECK-NEXT:      cf.cond_br %[[x]], ^[[for_1_body:.*]], ^[[for_1_out:.*]]
// CHECK-NEXT:  ^[[for_1_body]]:
// CHECK-NEXT:      cf.br ^[[for_2_condition:.*]]
// CHECK-NEXT:  ^[[for_2_condition]]:
// CHECK-NEXT:      cf.cond_br %[[y]], ^[[for_2_body:.*]], ^[[for_2_out:.*]]
// CHECK-NEXT:  ^[[for_2_body]]:
// CHECK-NEXT:      cf.cond_br %[[z]], ^[[if_true:.*]], ^[[if_out:.*]]
// CHECK-NEXT:  ^[[if_true]]:
// CHECK-NEXT:      cf.br ^[[for_2_out]]
// CHECK-NEXT:  ^[[if_out]]:
// CHECK-NEXT:      bmodelica.print %[[z]]
// CHECK-NEXT:      cf.br ^[[for_2_step:.*]]
// CHECK-NEXT:  ^[[for_2_step]]:
// CHECK-NEXT:      bmodelica.print %[[t]]
// CHECK-NEXT:      cf.br ^[[for_2_condition]]
// CHECK-NEXT:  ^[[for_2_out]]:
// CHECK-NEXT:      bmodelica.print %[[y]]
// CHECK-NEXT:      cf.br ^[[for_1_step:.*]]
// CHECK-NEXT:  ^[[for_1_step]]:
// CHECK-NEXT:      bmodelica.print %[[k]]
// CHECK-NEXT:      cf.br ^[[for_1_condition]]
// CHECK-NEXT:  ^[[for_1_out]]:
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<i1, input>
    bmodelica.variable @y : !bmodelica.variable<i1, input>
    bmodelica.variable @z : !bmodelica.variable<i1, input>
    bmodelica.variable @t : !bmodelica.variable<i64, input>
    bmodelica.variable @k : !bmodelica.variable<i64, input>
    bmodelica.variable @l : !bmodelica.variable<i64, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i1

        bmodelica.for condition {
            bmodelica.condition (%0 : i1)
        } body {
            %1 = bmodelica.variable_get @y : i1

            bmodelica.for condition {
                bmodelica.condition (%1 : i1)
            } body {
                %2 = bmodelica.variable_get @z : i1

                bmodelica.if (%2 : i1) {
                    bmodelica.break
                }

                bmodelica.print %2 : i1
                bmodelica.yield
            } step {
                %2 = bmodelica.variable_get @t : i64
                bmodelica.print %2 : i64
                bmodelica.yield
            }

            bmodelica.print %1 : i1
            bmodelica.yield
        } step {
            %1 = bmodelica.variable_get @k : i64
            bmodelica.print %1 : i64
            bmodelica.yield
        }

        bmodelica.print %0 : i1
    }
}
