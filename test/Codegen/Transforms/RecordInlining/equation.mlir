// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.real
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.real
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>

    modelica.equation {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.component_get %0, @x : !modelica.record<@R> -> !modelica.real
        %2 = modelica.variable_get @r : !modelica.record<@R>
        %3 = modelica.component_get %2, @y : !modelica.record<@R> -> !modelica.real
        %4 = modelica.equation_side %1 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}

// -----

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.real
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.real
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>

    modelica.equation {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.component_get %0, @x : !modelica.record<@R> -> !modelica.real
        %2 = modelica.component_get %0, @y : !modelica.record<@R> -> !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}

// -----

// Equality between two records.

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r1 : !modelica.variable<!modelica.record<@R>>
    modelica.variable @r2 : !modelica.variable<!modelica.record<@R>>

    modelica.equation {
        %0 = modelica.variable_get @r1 : !modelica.record<@R>
        %1 = modelica.variable_get @r2 : !modelica.record<@R>
        %2 = modelica.equation_side %0 : tuple<!modelica.record<@R>>
        %3 = modelica.equation_side %1 : tuple<!modelica.record<@R>>
        modelica.equation_sides %2, %3 : tuple<!modelica.record<@R>>, tuple<!modelica.record<@R>>
    }
}
