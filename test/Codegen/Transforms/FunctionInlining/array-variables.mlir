// RUN: modelica-opt %s --split-input-file --inline-functions | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-DAG:      %[[a:.*]] = modelica.variable_get @a
// CHECK-DAG:      %[[b:.*]] = modelica.variable_get @b
// CHECK-DAG:      %[[add:.*]] = modelica.add %[[a]], %[[b]]
// CHECK-DAG:      %[[c:.*]] = modelica.variable_get @c
// CHECK-DAG:      %[[lhs:.*]] = modelica.equation_side %[[add]]
// CHECK-DAG:      %[[rhs:.*]] = modelica.equation_side %[[c]]
// CHECK-NEXT:     modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK:       }

modelica.function @Foo attributes {inline = true} {
    modelica.variable @x : !modelica.variable<3x!modelica.real, input>
    modelica.variable @y : !modelica.variable<3x!modelica.real, input>
    modelica.variable @z : !modelica.variable<3x!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<3x!modelica.real>
        %2 = modelica.add %0, %1 : (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.real>) -> !modelica.array<3x!modelica.real>
        modelica.variable_set @z, %2 : !modelica.array<3x!modelica.real>
    }
}

modelica.model @Test {
    modelica.variable @a : !modelica.variable<3x!modelica.real>
    modelica.variable @b : !modelica.variable<3x!modelica.real>
    modelica.variable @c : !modelica.variable<3x!modelica.real>

    modelica.main_model {
        modelica.equation {
            %0 = modelica.variable_get @a : !modelica.array<3x!modelica.real>
            %1 = modelica.variable_get @b : !modelica.array<3x!modelica.real>
            %2 = modelica.call @Foo(%0, %1) : (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.real>) -> !modelica.array<3x!modelica.real>
            %3 = modelica.variable_get @c : !modelica.array<3x!modelica.real>
            %4 = modelica.equation_side %2 : tuple<!modelica.array<3x!modelica.real>>
            %5 = modelica.equation_side %3 : tuple<!modelica.array<3x!modelica.real>>
            modelica.equation_sides %4, %5 : tuple<!modelica.array<3x!modelica.real>>, tuple<!modelica.array<3x!modelica.real>>
        }
    }
}
