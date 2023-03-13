// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Binding equation for a scalar variable.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}

// -----

// Binding equation for an array variable.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[x]][%[[i0:.*]]]
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[array]][%[[i0]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.constant #modelica.int<1> : !modelica.int
      %2 = modelica.constant #modelica.int<2> : !modelica.int
      %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %3 : !modelica.array<3x!modelica.int>
    }
}

// -----

// Binding equation for a scalar parameter.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int, parameter>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}

// -----

// Binding equation for an array parameter.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[value:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x!modelica.int, parameter>

    modelica.binding_equation @x {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.constant #modelica.int<1> : !modelica.int
      %2 = modelica.constant #modelica.int<2> : !modelica.int
      %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %3 : !modelica.array<3x!modelica.int>
    }
}
