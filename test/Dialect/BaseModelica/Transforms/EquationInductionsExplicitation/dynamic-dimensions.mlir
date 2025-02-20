// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

bmodelica.function @foo {
    bmodelica.variable @in : !bmodelica.variable<3x4x5x!bmodelica.int, input>
    bmodelica.variable @out : !bmodelica.variable<3x?x5x!bmodelica.int, output>
}

bmodelica.function @bar {
    bmodelica.variable @in : !bmodelica.variable<3x4x5x!bmodelica.int, input>
    bmodelica.variable @out : !bmodelica.variable<3x4x?x!bmodelica.int, output>
}

// CHECK-LABEL: @array3d

bmodelica.model @array3d {
    bmodelica.variable @x : !bmodelica.variable<3x4x5x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x4x5x!bmodelica.int>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : tensor<3x4x5x!bmodelica.int>
            %1 = bmodelica.call @foo(%0) : (tensor<3x4x5x!bmodelica.int>) -> tensor<3x?x5x!bmodelica.int>
            %2 = bmodelica.variable_get @y : tensor<3x4x5x!bmodelica.int>
            %3 = bmodelica.call @bar(%2) : (tensor<3x4x5x!bmodelica.int>) -> tensor<3x4x?x!bmodelica.int>
            %4 = bmodelica.equation_side %1 : tuple<tensor<3x?x5x!bmodelica.int>>
            %5 = bmodelica.equation_side %3 : tuple<tensor<3x4x?x!bmodelica.int>>
            bmodelica.equation_sides %4, %5 : tuple<tensor<3x?x5x!bmodelica.int>>, tuple<tensor<3x4x?x!bmodelica.int>>
        }

        // CHECK:       bmodelica.for_equation %[[i0:.*]] = 0 to 2
        // CHECK-NEXT:      bmodelica.for_equation %[[i1:.*]] = 0 to 3
        // CHECK-NEXT:          bmodelica.for_equation %[[i2:.*]] = 0 to 4
        // CHECK-DAG:               %[[x:.*]] = bmodelica.variable_get @x
        // CHECK-DAG:               %[[y:.*]] = bmodelica.variable_get @y
        // CHECK-DAG:               %[[call_foo:.*]] = bmodelica.call @foo(%[[x]])
        // CHECK-DAG:               %[[call_bar:.*]] = bmodelica.call @bar(%[[y]])
        // CHECK-DAG:               %[[lhs_extract:.*]] = bmodelica.tensor_extract %[[call_foo]][%[[i0]], %[[i1]], %[[i2]]]
        // CHECK-DAG:               %[[rhs_extract:.*]] = bmodelica.tensor_extract %[[call_bar]][%[[i0]], %[[i1]], %[[i2]]]
        // CHECK-DAG:               %[[lhs:.*]] = bmodelica.equation_side %[[lhs_extract]]
        // CHECK-DAG:               %[[rhs:.*]] = bmodelica.equation_side %[[rhs_extract]]
        // CHECK:                   bmodelica.equation_sides %[[lhs]], %[[rhs]]
    }
}
