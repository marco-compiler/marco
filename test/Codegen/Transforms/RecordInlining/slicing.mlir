// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<5x3x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<5x3x!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<5x3x!bmodelica.real>
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<5x3x!bmodelica.real>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[y]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>
}

bmodelica.model @Test {
    bmodelica.variable @r : !bmodelica.variable<5x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica.array<5x!bmodelica<record @R>>
            %1 = bmodelica.component_get %0, @x : !bmodelica.array<5x!bmodelica<record @R>> -> !bmodelica.array<5x3x!bmodelica.real>
            %2 = bmodelica.component_get %0, @y : !bmodelica.array<5x!bmodelica<record @R>> -> !bmodelica.array<5x3x!bmodelica.real>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.array<5x3x!bmodelica.real>>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.array<5x3x!bmodelica.real>>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.array<5x3x!bmodelica.real>>, tuple<!bmodelica.array<5x3x!bmodelica.real>>
        }
    }
}
