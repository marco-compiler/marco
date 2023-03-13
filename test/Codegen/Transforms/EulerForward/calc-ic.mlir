// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(euler-forward{model=Test})" | FileCheck %s

// CHECK:       simulation.function @calcIC(solvers: [], time: [%[[time:.*]]: !modelica.real], variables: [], extra_args: []) -> () {
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
