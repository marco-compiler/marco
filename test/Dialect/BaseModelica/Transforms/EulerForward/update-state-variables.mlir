// RUN: modelica-opt %s --split-input-file --euler-forward | FileCheck %s

// CHECK:       runtime.function @updateStateVariables(%[[timeStep:.*]]: f64) {
// CHECK:           runtime.return
// CHECK-NEXT:  }

bmodelica.model @emptyModel {

}
