// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(euler-forward{model=Test})" | FileCheck %s

// CHECK:       simulation.function @updateStateVariables(solvers: [], time: [%[[time:.*]]: !modelica.real], variables: [], extra_args: [%[[timeStep:.*]]: f64]) -> () {
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
