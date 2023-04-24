// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// Input record.

// CHECK-LABEL: @R
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real>

// CHECK-LABEL: @Foo
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real, input>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real, input>
// CHECK: modelica.variable @s : !modelica.variable<!modelica.real, output>

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.real
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.real
// CHECK:           %[[call:.*]] = modelica.call @Foo(%[[x]], %[[y]]) : (!modelica.real, !modelica.real) -> !modelica.real
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[call]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Foo {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>, input>
    modelica.variable @s : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.component_get %0, @x : !modelica.record<@R> -> !modelica.real
        %2 = modelica.component_get %0, @y : !modelica.record<@R> -> !modelica.real
        %3 = modelica.add %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @s, %3 : !modelica.real
    }
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>

    modelica.equation {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.call @Foo(%0) : (!modelica.record<@R>) -> !modelica.real
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}

// -----

// Output record.

// CHECK-LABEL: @R
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real>

// CHECK-LABEL: @Foo
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real, output>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real, output>

// CHECK-LABEL: @Test
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x : !modelica.real
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @y : !modelica.real
// CHECK:           %[[call:.*]]:2 = modelica.call @Foo(%[[x]], %[[y]]) : (!modelica.real, !modelica.real) -> (!modelica.real, !modelica.real)
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[call]]#0
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[call]]#1
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Foo {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @r : !modelica.variable<!modelica.record<@R>, output>

    modelica.algorithm {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.variable_get @x : !modelica.real
        %2 = modelica.variable_get @y : !modelica.real
        modelica.component_set %0, @x, %1 : !modelica.record<@R>, !modelica.real
        modelica.component_set %0, @y, %2 : !modelica.record<@R>, !modelica.real
    }
}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>

    modelica.equation {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.call @Foo(%0, %1) : (!modelica.real, !modelica.real) -> !modelica.record<@R>
        %3 = modelica.component_get %2, @x : !modelica.record<@R> -> !modelica.real
        %4 = modelica.component_get %2, @y : !modelica.record<@R> -> !modelica.real
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}
