// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

// CHECK:       bmodelica.model @Test
// CHECK:       bmodelica.for_equation %[[i0:.*]] = 0 to 2 {
// CHECK-NEXT:      bmodelica.for_equation %[[i1:.*]] = 0 to 3 {
// CHECK-NEXT:          bmodelica.for_equation %[[i2:.*]] = 0 to 4 {
// CHECK-DAG:               %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:               %[[y:.*]] = bmodelica.variable_get @y
// CHECK-DAG:               %[[call_foo:.*]] = bmodelica.call @foo(%[[x]])
// CHECK-DAG:               %[[call_bar:.*]] = bmodelica.call @bar(%[[y]])
// CHECK-DAG:               %[[lhs_load:.*]] = bmodelica.load %[[call_foo]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[rhs_load:.*]] = bmodelica.load %[[call_bar]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[lhs:.*]] = bmodelica.equation_side %[[lhs_load]]
// CHECK-DAG:               %[[rhs:.*]] = bmodelica.equation_side %[[rhs_load]]
// CHECK:                   bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    bmodelica.function @foo {
        bmodelica.variable @in : !bmodelica.variable<3x4x5x!bmodelica.int, input>
        bmodelica.variable @out : !bmodelica.variable<3x?x5x!bmodelica.int, output>
    }

    bmodelica.function @bar {
        bmodelica.variable @in : !bmodelica.variable<3x4x5x!bmodelica.int, input>
        bmodelica.variable @out : !bmodelica.variable<3x4x?x!bmodelica.int, output>
    }

    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<3x4x5x!bmodelica.int>
        bmodelica.variable @y : !bmodelica.variable<3x4x5x!bmodelica.int>

        bmodelica.dynamic {
            bmodelica.equation {
                %0 = bmodelica.variable_get @x : !bmodelica.array<3x4x5x!bmodelica.int>
                %1 = bmodelica.call @foo(%0) : (!bmodelica.array<3x4x5x!bmodelica.int>) -> !bmodelica.array<3x?x5x!bmodelica.int>
                %2 = bmodelica.variable_get @y : !bmodelica.array<3x4x5x!bmodelica.int>
                %3 = bmodelica.call @bar(%2) : (!bmodelica.array<3x4x5x!bmodelica.int>) -> !bmodelica.array<3x4x?x!bmodelica.int>
                %4 = bmodelica.equation_side %1 : tuple<!bmodelica.array<3x?x5x!bmodelica.int>>
                %5 = bmodelica.equation_side %3 : tuple<!bmodelica.array<3x4x?x!bmodelica.int>>
                bmodelica.equation_sides %4, %5 : tuple<!bmodelica.array<3x?x5x!bmodelica.int>>, tuple<!bmodelica.array<3x4x?x!bmodelica.int>>
            }
        }
    }
}
