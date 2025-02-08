// RUN: modelica-opt %s --split-input-file --euler-forward | FileCheck %s

// CHECK:       runtime.function @updateNonStateVariables() {
// CHECK:           runtime.return
// CHECK-NEXT:  }

bmodelica.model @emptyModel {

}
