// RUN: modelica-opt %s --split-input-file --pure-function-inlining | FileCheck %s

bmodelica.function @Foo attributes {inline = true} {
    bmodelica.variable @x : !bmodelica.variable<3xf64, input>
    bmodelica.variable @y : !bmodelica.variable<3xf64, input>
    bmodelica.variable @z : !bmodelica.variable<3xf64, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3xf64>
        %1 = bmodelica.variable_get @y : tensor<3xf64>
        %2 = bmodelica.add %0, %1 : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
        bmodelica.variable_set @z, %2 : tensor<3xf64>
    }
}

// CHECK-LABEL: @arrayVariables

bmodelica.model @arrayVariables {
    bmodelica.variable @a : !bmodelica.variable<3xf64>
    bmodelica.variable @b : !bmodelica.variable<3xf64>
    bmodelica.variable @c : !bmodelica.variable<3xf64>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @a : tensor<3xf64>
            %1 = bmodelica.variable_get @b : tensor<3xf64>
            %2 = bmodelica.call @Foo(%0, %1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
            %3 = bmodelica.variable_get @c : tensor<3xf64>
            %4 = bmodelica.equation_side %2 : tuple<tensor<3xf64>>
            %5 = bmodelica.equation_side %3 : tuple<tensor<3xf64>>
            bmodelica.equation_sides %4, %5 : tuple<tensor<3xf64>>, tuple<tensor<3xf64>>

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
