// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// Scalar variable.

// CHECK:       bmodelica.global_variable @[[var:.*]] : !bmodelica.array<!bmodelica.int>
// CHECK:       runtime.variable_getter @[[getter:.*]]() -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.global_variable_get @[[var]]
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.int -> f64
// CHECK-NEXT:      runtime.return %[[cast]]
// CHECK-NEXT:  }
// CHECK:       runtime.variable_getters [@[[getter]]]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    }
}

// -----

// Array variable.

// CHECK:       bmodelica.global_variable @[[var:.*]] : !bmodelica.array<2x3x!bmodelica.int>
// CHECK:       runtime.variable_getter @[[getter:.*]](%[[i0:.*]]: index, %[[i1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.global_variable_get @[[var]]
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][%[[i0]], %[[i1]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.int -> f64
// CHECK-NEXT:      runtime.return %[[cast]]
// CHECK-NEXT:  }
// CHECK:       runtime.variable_getters [@[[getter]]]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.int>
    }
}

// -----

// CHECK-DAG: runtime.variable_getter @[[getter0:.*]]() -> f64
// CHECK-DAG: runtime.variable_getter @[[getter1:.*]](%{{.*}}: index, %{{.*}}: index) -> f64

// CHECK-DAG: runtime.variable_getters [@[[getter0]], @[[getter1]]]

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
        bmodelica.variable @y : !bmodelica.variable<2x3x!bmodelica.int>
    }
}
