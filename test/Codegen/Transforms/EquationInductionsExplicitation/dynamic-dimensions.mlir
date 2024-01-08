// RUN: modelica-opt %s --split-input-file --explicitate-equation-inductions | FileCheck %s

// CHECK:       modelica.model @Test
// CHECK:       modelica.for_equation %[[i0:.*]] = 0 to 2 {
// CHECK-NEXT:      modelica.for_equation %[[i1:.*]] = 0 to 3 {
// CHECK-NEXT:          modelica.for_equation %[[i2:.*]] = 0 to 4 {
// CHECK-DAG:               %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:               %[[y:.*]] = modelica.variable_get @y
// CHECK-DAG:               %[[call_foo:.*]] = modelica.call @foo(%[[x]])
// CHECK-DAG:               %[[call_bar:.*]] = modelica.call @bar(%[[y]])
// CHECK-DAG:               %[[lhs_load:.*]] = modelica.load %[[call_foo]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[rhs_load:.*]] = modelica.load %[[call_bar]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK-DAG:               %[[lhs:.*]] = modelica.equation_side %[[lhs_load]]
// CHECK-DAG:               %[[rhs:.*]] = modelica.equation_side %[[rhs_load]]
// CHECK:                   modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    modelica.function @foo {
        modelica.variable @in : !modelica.variable<3x4x5x!modelica.int, input>
        modelica.variable @out : !modelica.variable<3x?x5x!modelica.int, output>
    }

    modelica.function @bar {
        modelica.variable @in : !modelica.variable<3x4x5x!modelica.int, input>
        modelica.variable @out : !modelica.variable<3x4x?x!modelica.int, output>
    }

    modelica.model @Test {
        modelica.variable @x : !modelica.variable<3x4x5x!modelica.int>
        modelica.variable @y : !modelica.variable<3x4x5x!modelica.int>

        modelica.equation {
            %0 = modelica.variable_get @x : !modelica.array<3x4x5x!modelica.int>
            %1 = modelica.call @foo(%0) : (!modelica.array<3x4x5x!modelica.int>) -> !modelica.array<3x?x5x!modelica.int>
            %2 = modelica.variable_get @y : !modelica.array<3x4x5x!modelica.int>
            %3 = modelica.call @bar(%2) : (!modelica.array<3x4x5x!modelica.int>) -> !modelica.array<3x4x?x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.array<3x?x5x!modelica.int>>
            %5 = modelica.equation_side %3 : tuple<!modelica.array<3x4x?x!modelica.int>>
            modelica.equation_sides %4, %5 : tuple<!modelica.array<3x?x5x!modelica.int>>, tuple<!modelica.array<3x4x?x!modelica.int>>
        }
    }
}
