// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

// CHECK-LABEL: @array3d

bmodelica.model @array3d {
    bmodelica.variable @x : !bmodelica.variable<3x4x5x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x4x5x!bmodelica.int>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : tensor<3x4x5x!bmodelica.int>
            %1 = bmodelica.variable_get @y : tensor<3x4x5x!bmodelica.int>
            %2 = bmodelica.equation_side %0 : tuple<tensor<3x4x5x!bmodelica.int>>
            %3 = bmodelica.equation_side %1 : tuple<tensor<3x4x5x!bmodelica.int>>
            bmodelica.equation_sides %2, %3 : tuple<tensor<3x4x5x!bmodelica.int>>, tuple<tensor<3x4x5x!bmodelica.int>>
        }

        // CHECK:       bmodelica.for_equation %[[i0:.*]] = 0 to 2
        // CHECK-NEXT:      bmodelica.for_equation %[[i1:.*]] = 0 to 3
        // CHECK-NEXT:          bmodelica.for_equation %[[i2:.*]] = 0 to 4
        // CHECK-DAG:               %[[x:.*]] = bmodelica.variable_get @x
        // CHECK-DAG:               %[[y:.*]] = bmodelica.variable_get @y
        // CHECK-DAG:               %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]], %[[i1]], %[[i2]]]
        // CHECK-DAG:               %[[y_extract:.*]] = bmodelica.tensor_extract %[[y]][%[[i0]], %[[i1]], %[[i2]]]
        // CHECK-DAG:               %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
        // CHECK-DAG:               %[[rhs:.*]] = bmodelica.equation_side %[[y_extract]]
        // CHECK:                   bmodelica.equation_sides %[[lhs]], %[[rhs]]
    }
}
