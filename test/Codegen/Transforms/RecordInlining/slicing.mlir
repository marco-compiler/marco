// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<5x3x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<5x3x!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<5x3x!modelica.real>
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<5x3x!modelica.real>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<3x!modelica.real>
    modelica.variable @y : !modelica.variable<3x!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<5x!modelica<record @R>>

    modelica.equation {
        %0 = modelica.variable_get @r : !modelica.array<5x!modelica<record @R>>
        %1 = modelica.component_get %0, @x : !modelica.array<5x!modelica<record @R>> -> !modelica.array<5x3x!modelica.real>
        %2 = modelica.component_get %0, @y : !modelica.array<5x!modelica<record @R>> -> !modelica.array<5x3x!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.array<5x3x!modelica.real>>
        %4 = modelica.equation_side %2 : tuple<!modelica.array<5x3x!modelica.real>>
        modelica.equation_sides %3, %4 : tuple<!modelica.array<5x3x!modelica.real>>, tuple<!modelica.array<5x3x!modelica.real>>
    }
}
