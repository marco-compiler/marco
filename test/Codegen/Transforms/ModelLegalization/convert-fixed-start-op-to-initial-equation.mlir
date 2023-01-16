// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// Scalar variable with fixed start value

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.initial_equation {
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.start (%arg0 : !modelica.array<!modelica.int>) {each = false, fixed = true} {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

// Array variable with fixed start scalar value

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>):
// CHECK:       modelica.for_equation %[[i:.*]] = 0 to 2 {
// CHECK:           modelica.initial_equation {
// CHECK-DAG:           %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-DAG:           %[[x_load:.*]] = modelica.load %[[x]][%[[i]]]
// CHECK-DAG:           %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:           %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK:               modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int>
    modelica.yield %0 : !modelica.member<3x!modelica.int>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>):
    modelica.start (%arg0 : !modelica.array<3x!modelica.int>) {each = true, fixed = true} {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

// Array variable with fixed start array value

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>):
// CHECK:       modelica.initial_equation {
// CHECK-DAG:       %[[value_0:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-DAG:       %[[value_1:.*]] = modelica.constant #modelica.int<1> : !modelica.int
// CHECK-DAG:       %[[value_2:.*]] = modelica.constant #modelica.int<2> : !modelica.int
// CHECK-DAG:       %[[value:.*]] = modelica.array_from_elements %[[value_0]], %[[value_1]], %[[value_2]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[value]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int>
    modelica.yield %0 : !modelica.member<3x!modelica.int>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>):
    modelica.start (%arg0 : !modelica.array<3x!modelica.int>) {each = false, fixed = true} {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        %1 = modelica.constant #modelica.int<1> : !modelica.int
        %2 = modelica.constant #modelica.int<2> : !modelica.int
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    }
}
