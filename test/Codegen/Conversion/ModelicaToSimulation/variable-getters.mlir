// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// Scalar variable.

// CHECK:       modelica.global_variable @[[var:.*]] : !modelica.array<!modelica.int>
// CHECK:       simulation.variable_getter @[[getter:.*]]() -> f64 {
// CHECK:           %[[get:.*]] = modelica.global_variable_get @[[var]]
// CHECK:           %[[load:.*]] = modelica.load %[[get]][]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.int -> f64
// CHECK-NEXT:      simulation.return %[[cast]]
// CHECK-NEXT:  }
// CHECK:       simulation.variable_getters [@[[getter]]]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
}

// -----

// Array variable.

// CHECK:       modelica.global_variable @[[var:.*]] : !modelica.array<2x3x!modelica.int>
// CHECK:       simulation.variable_getter @[[getter:.*]](%[[i0:.*]]: index, %[[i1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = modelica.global_variable_get @[[var]]
// CHECK:           %[[load:.*]] = modelica.load %[[get]][%[[i0]], %[[i1]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.int -> f64
// CHECK-NEXT:      simulation.return %[[cast]]
// CHECK-NEXT:  }
// CHECK:       simulation.variable_getters [@[[getter]]]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<2x3x!modelica.int>
    }

    modelica.simulation_variable @x : !modelica.variable<2x3x!modelica.int>
}

// -----

// CHECK-DAG: simulation.variable_getter @[[getter0:.*]]() -> f64
// CHECK-DAG: simulation.variable_getter @[[getter1:.*]](%{{.*}}: index, %{{.*}}: index) -> f64

// CHECK-DAG: simulation.variable_getters [@[[getter0]], @[[getter1]]]

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
        modelica.variable @y : !modelica.variable<2x3x!modelica.int>
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
    modelica.simulation_variable @y : !modelica.variable<2x3x!modelica.int>
}
