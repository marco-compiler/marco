// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

// CHECK:       modelica.model @Test
// CHECK:       modelica.for_equation %[[i0:.*]] = 0 to 2 {
// CHECK-NEXT:      modelica.for_equation %[[i1:.*]] = 0 to 3 {
// CHECK-NEXT:          modelica.for_equation %[[i2:.*]] = 0 to 4 {
// CHECK-DAG:               %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:               %[[y:.*]] = modelica.variable_get @y
// CHECK-DAG:               %[[x_load:.*]] = modelica.load %[[x]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[y_load:.*]] = modelica.load %[[y]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:               %[[rhs:.*]] = modelica.equation_side %[[y_load]]
// CHECK:                   modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x4x5x!modelica.int>
    modelica.variable @y : !modelica.variable<3x4x5x!modelica.int>

    modelica.equation {
        %0 = modelica.variable_get @x : !modelica.array<3x4x5x!modelica.int>
        %1 = modelica.variable_get @y : !modelica.array<3x4x5x!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.array<3x4x5x!modelica.int>>
        %3 = modelica.equation_side %1 : tuple<!modelica.array<3x4x5x!modelica.int>>
        modelica.equation_sides %2, %3 : tuple<!modelica.array<3x4x5x!modelica.int>>, tuple<!modelica.array<3x4x5x!modelica.int>>
    }
}
