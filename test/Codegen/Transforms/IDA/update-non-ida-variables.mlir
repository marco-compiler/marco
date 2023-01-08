// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// CHECK:       simulation.function @updateNonIDAVariables(solvers: [%[[ida:.*]]: !ida.instance], time: [%[[time:.*]]: !modelica.real], variables: [], extra_args: []) -> () {
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.yield
} body {

}
