// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

// CHECK:       bmodelica.model @Test
// CHECK:       bmodelica.for_equation %[[i0:.*]] = 0 to 2 {
// CHECK-NEXT:      bmodelica.for_equation %[[i1:.*]] = 0 to 3 {
// CHECK-NEXT:          bmodelica.for_equation %[[i2:.*]] = 0 to 4 {
// CHECK-DAG:               %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:               %[[y:.*]] = bmodelica.variable_get @y
// CHECK-DAG:               %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[y_load:.*]] = bmodelica.load %[[y]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
// CHECK-DAG:               %[[rhs:.*]] = bmodelica.equation_side %[[y_load]]
// CHECK:                   bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x4x5x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x4x5x!bmodelica.int>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.array<3x4x5x!bmodelica.int>
            %1 = bmodelica.variable_get @y : !bmodelica.array<3x4x5x!bmodelica.int>
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.array<3x4x5x!bmodelica.int>>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.array<3x4x5x!bmodelica.int>>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.array<3x4x5x!bmodelica.int>>, tuple<!bmodelica.array<3x4x5x!bmodelica.int>>
        }
    }
}
