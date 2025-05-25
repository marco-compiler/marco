// RUN: modelica-opt %s --split-input-file --pure-function-inlining | FileCheck %s

bmodelica.function @Foo attributes {inline = true} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// CHECK-LABEL: @scalarVariables

bmodelica.model @scalarVariables {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @c : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @a : !bmodelica.real
            %1 = bmodelica.variable_get @b : !bmodelica.real
            %2 = bmodelica.call @Foo(%0, %1) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            %3 = bmodelica.variable_get @c : !bmodelica.real
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[a:.*]] = bmodelica.variable_get @a
            // CHECK-DAG:   %[[b:.*]] = bmodelica.variable_get @b
            // CHECK-DAG:   %[[add:.*]] = bmodelica.add %[[a]], %[[b]]
            // CHECK-DAG:   %[[c:.*]] = bmodelica.variable_get @c
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[add]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[c]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}
