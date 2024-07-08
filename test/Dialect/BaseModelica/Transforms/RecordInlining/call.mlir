// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// Input record.

// CHECK-LABEL: @R
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

// CHECK-LABEL: @Foo
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real, input>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real, input>
// CHECK: bmodelica.variable @s : !bmodelica.variable<!bmodelica.real, output>

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.real
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.real
// CHECK:           %[[call:.*]] = bmodelica.call @Foo(%[[x]], %[[y]]) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[call]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, input>
    bmodelica.variable @s : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @r : !bmodelica<record @R>
        %1 = bmodelica.component_get %0, @x : !bmodelica<record @R> -> !bmodelica.real
        %2 = bmodelica.component_get %0, @y : !bmodelica<record @R> -> !bmodelica.real
        %3 = bmodelica.add %1, %2 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @s, %3 : !bmodelica.real
    }
}

bmodelica.model @Test {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica<record @R>
            %1 = bmodelica.call @Foo(%0) : (!bmodelica<record @R>) -> !bmodelica.real
            %2 = bmodelica.constant #bmodelica<real 0.0>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}

// -----

// Output record.

// CHECK-LABEL: @R
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

// CHECK-LABEL: @Foo
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real, output>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real, output>

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.real
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y : !bmodelica.real
// CHECK:           %[[call:.*]]:2 = bmodelica.call @Foo(%[[x]], %[[y]]) : (!bmodelica.real, !bmodelica.real) -> (!bmodelica.real, !bmodelica.real)
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[call]]#0
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[call]]#1
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        bmodelica.variable_component_set @r::@x, %0 : !bmodelica.real
        bmodelica.variable_component_set @r::@y, %1 : !bmodelica.real
    }
}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.variable_get @y : !bmodelica.real
            %2 = bmodelica.call @Foo(%0, %1) : (!bmodelica.real, !bmodelica.real) -> !bmodelica<record @R>
            %3 = bmodelica.component_get %2, @x : !bmodelica<record @R> -> !bmodelica.real
            %4 = bmodelica.component_get %2, @y : !bmodelica<record @R> -> !bmodelica.real
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}
