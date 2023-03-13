// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Scalar variable with fixed start value.

// CHECK:       modelica.initial_equation {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = false, fixed = true}
}

// -----

// Array variable with fixed start scalar value.

// CHECK:       modelica.for_equation %[[i:.*]] = 0 to 2 {
// CHECK:           modelica.initial_equation {
// CHECK-NEXT:          %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:          %[[x_load:.*]] = modelica.load %[[x]][%[[i]]]
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:          %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-NEXT:          %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK-NEXT:          modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = true, fixed = true}
}

// -----

// Array variable with fixed start array value

// CHECK:       modelica.initial_equation {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[value_0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[value_1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[value_2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[value:.*]] = modelica.array_from_elements %[[value_0]], %[[value_1]], %[[value_2]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.start @x {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        %1 = modelica.constant #modelica.int<1> : !modelica.int
        %2 = modelica.constant #modelica.int<2> : !modelica.int
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    } {each = false, fixed = true}
}
