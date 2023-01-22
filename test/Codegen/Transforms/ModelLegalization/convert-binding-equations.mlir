// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Binding equation for a scalar variable.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[x]][]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<!modelica.int>) {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}

// -----

// Binding equation for an array variable.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>):
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[x]][%[[i0:.*]]]
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %4[%[[i0]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int>
    modelica.yield %0 : !modelica.member<3x!modelica.int>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<3x!modelica.int>) {
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
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.start (%[[x]] : !modelica.array<!modelica.int>) {each = false, fixed = true} {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, parameter>
    modelica.yield %0 : !modelica.member<!modelica.int, parameter>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<!modelica.int>) {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
}

// -----

// Binding equation for an array parameter.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>):
// CHECK:       modelica.start (%[[x]] : !modelica.array<3x!modelica.int>) {each = false, fixed = true} {
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[value:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int, parameter>
    modelica.yield %0 : !modelica.member<3x!modelica.int, parameter>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<3x!modelica.int>) {
      %0 = modelica.constant #modelica.int<0> : !modelica.int
      %1 = modelica.constant #modelica.int<1> : !modelica.int
      %2 = modelica.constant #modelica.int<2> : !modelica.int
      %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
      modelica.yield %3 : !modelica.array<3x!modelica.int>
    }
}
