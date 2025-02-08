// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// COM: Input record.

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @inputRecordFunction

bmodelica.function @inputRecordFunction {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, input>
    bmodelica.variable @s : !bmodelica.variable<!bmodelica.real, output>

    // CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real, input>
    // CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @r : !bmodelica<record @R>
        %1 = bmodelica.component_get %0, @x : !bmodelica<record @R> -> !bmodelica.real
        %2 = bmodelica.component_get %0, @y : !bmodelica<record @R> -> !bmodelica.real
        %3 = bmodelica.add %1, %2 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @s, %3 : !bmodelica.real
    }
}

// CHECK-LABEL: @inputRecordModel

bmodelica.model @inputRecordModel {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica<record @R>
            %1 = bmodelica.call @inputRecordFunction(%0) : (!bmodelica<record @R>) -> !bmodelica.real

            // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.real
            // CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.real
            // CHECK:           %[[call:.*]] = bmodelica.call @inputRecordFunction(%[[x]], %[[y]]) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real

            %2 = bmodelica.constant #bmodelica<real 0.0>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}

// -----

// COM: Output record.

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @outputRecordFunction

bmodelica.function @outputRecordFunction {
    bmodelica.variable @v : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, output>

    // CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real, output>
    // CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @v : !bmodelica.real
        bmodelica.variable_component_set @r::@x, %0 : !bmodelica.real
        bmodelica.variable_component_set @r::@y, %0 : !bmodelica.real
    }
}

// CHECK-LABEL: @outputRecordModel

bmodelica.model @outputRecordModel {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.call @outputRecordFunction(%0) : (!bmodelica.real) -> !bmodelica<record @R>
            %2 = bmodelica.component_get %1, @x : !bmodelica<record @R> -> !bmodelica.real
            %3 = bmodelica.component_get %1, @y : !bmodelica<record @R> -> !bmodelica.real
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK:       %[[call:.*]]:2 = bmodelica.call @outputRecordFunction(%{{.*}}) : (!bmodelica.real) -> (!bmodelica.real, !bmodelica.real)
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[call]]#0
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[call]]#1
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}
