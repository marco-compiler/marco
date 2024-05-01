// RUN: modelica-opt %s --split-input-file --inline-functions | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-DAG:      %[[a:.*]] = bmodelica.variable_get @a
// CHECK-DAG:      %[[b:.*]] = bmodelica.variable_get @b
// CHECK-DAG:      %[[add:.*]] = bmodelica.add %[[a]], %[[b]]
// CHECK-DAG:      %[[c:.*]] = bmodelica.variable_get @c
// CHECK-DAG:      %[[lhs:.*]] = bmodelica.equation_side %[[add]]
// CHECK-DAG:      %[[rhs:.*]] = bmodelica.equation_side %[[c]]
// CHECK-NEXT:     bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK:       }

bmodelica.function @Foo attributes {inline = true} {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.real>
        %1 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.real>
        %2 = bmodelica.add %0, %1 : (!bmodelica.array<3x!bmodelica.real>, !bmodelica.array<3x!bmodelica.real>) -> !bmodelica.array<3x!bmodelica.real>
        bmodelica.variable_set @z, %2 : !bmodelica.array<3x!bmodelica.real>
    }
}

bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @b : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @c : !bmodelica.variable<3x!bmodelica.real>

    bmodelica.main_model {
        bmodelica.equation {
            %0 = bmodelica.variable_get @a : !bmodelica.array<3x!bmodelica.real>
            %1 = bmodelica.variable_get @b : !bmodelica.array<3x!bmodelica.real>
            %2 = bmodelica.call @Foo(%0, %1) : (!bmodelica.array<3x!bmodelica.real>, !bmodelica.array<3x!bmodelica.real>) -> !bmodelica.array<3x!bmodelica.real>
            %3 = bmodelica.variable_get @c : !bmodelica.array<3x!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.array<3x!bmodelica.real>>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.array<3x!bmodelica.real>>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.array<3x!bmodelica.real>>, tuple<!bmodelica.array<3x!bmodelica.real>>
        }
    }
}
