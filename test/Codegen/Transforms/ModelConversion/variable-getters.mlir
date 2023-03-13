// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// Scalar variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: name = "x"

// CHECK: simulation.module
// CHECK:       simulation.variable_getter [#[[x]]](%[[variable:.*]]: !modelica.array<!modelica.real>) -> !modelica.real {
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[variable]][]
// CHECK-NEXT:      simulation.yield %[[result]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>
}

// -----

// Array variable

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: name = "x"

// CHECK: simulation.module
// CHECK:       simulation.variable_getter [#[[x]]](%[[variable:.*]]: !modelica.array<3x2x!modelica.real>, %[[i0:.*]]: index, %[[i1:.*]]: index) -> !modelica.real {
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[variable]][%[[i0]], %[[i1]]]
// CHECK-NEXT:      simulation.yield %[[result]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<3x2x!modelica.real>
}

// -----

// Multiple variables with the same type

// CHECK: #[[x:.*]] = #simulation.variable
// CHECK-SAME: name = "x"

// CHECK: #[[y:.*]] = #simulation.variable
// CHECK-SAME: name = "y"

// CHECK: simulation.module
// CHECK:       simulation.variable_getter [#[[x]], #[[y]]](%[[variable:.*]]: !modelica.array<!modelica.real>) -> !modelica.real {
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[variable]][]
// CHECK-NEXT:      simulation.yield %[[result]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>
    modelica.variable @y : !modelica.member<!modelica.real>
}
